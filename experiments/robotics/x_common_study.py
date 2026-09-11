"""Common cylindrical identification, held-out prediction, and fresh X plans.

Run prepare only after both pilot fits pass. Source and inputs are frozen before
planning. Each shaping rollout uses a fresh specimen of the calibrated material;
it is not a continuation from the plastically deformed calibration specimen.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyvista as pv

from experiments.robotics import x_common_probe as probe
from experiments.robotics import x_letter_study as position
from experiments.robotics import x_work_control as control
from experiments.robotics import x_work_study as work
from experiments.robotics.plastic_shaping_study import TRUTH, save_json, save_npz

ROOT = position.ROOT
OUT = ROOT / "out/x_common_identification_study_20260910"
PILOT = ROOT / "out/x_common_probe32_pilot_20260910"
ROBOT = ROOT / "out/x_work_guard12_study_20260909/franka/panda_model_snapshot"
digest = position.digest
control.MIN_GAP = work.MIN_GAP = .012
position.PROTOCOL = dict(position.PROTOCOL, models=["identified_A", "identified_B"],
    identification="Same prescribed cylindrical squeeze on both; synthetic x, v, elastic F; no force or stress labels",
    execution="Fresh specimen of the same material; selected model gaps converted to work and both work vectors cross-executed")
PROTOCOL = dict(scene=probe.core.CONFIG, probe=probe.PROBE, validation=probe.VALIDATION,
    identification=dict(frame_stride=2, window_frames=26, contact_margin_cells=2.,
        channels=["x", "v", "elastic F", "mass", "reference volume"],
        known=["geometry", "density", "contact", "Hencky/von-Mises family"],
        truth_used=False, force_used=False, yield_activity="Independent simulator audit; not an estimator input"),
    search=position.PROTOCOL, settings=position.SETTINGS, minimum_gap_m=.012,
    command="Four positive compression-work targets of both fingers; 50 mm/s closing; 4 ms feedback; common 12 mm guard",
    specimens="Identification, held-out validation and shaping each start from a fresh identical block of the same material",
    target="Unchanged rounded X and common initialization from earlier A feasibility; no retuning on new true-material results",
    scope="Synthetic supplied-state identification and contact simulation; no hardware, camera-state recovery, constitutive-family selection or arm dynamics")


def prepare(folder, pilot):
    assert not folder.exists(), "Use a fresh archive"
    models = {f"true_{n}": law for n, law in TRUTH.items()}
    for n in "AB":
        fit = json.loads((pilot / f"identification_{n}.json").read_text())
        excitation = json.loads((pilot / f"excitation_{n}.json").read_text())
        config = json.loads((pilot / f"probe_{n}.config.json").read_text())
        assert config["protocol"] == probe.PROBE
        assert fit["accepted"] and not fit["plastic"]["refused"]
        assert excitation["outgoing_trial_over_true_cap_fraction"] > .01
        models[f"identified_{n}"] = fit["law"]
    folder.mkdir(parents=True)
    inputs = folder / "inputs"
    inputs.mkdir()
    for n in "AB":
        for stem in [f"probe_{n}.npz", f"probe_{n}.config.json", f"probe_{n}.phases.json",
                     f"identification_{n}.json", f"excitation_{n}.json"]:
            shutil.copy2(pilot / stem, inputs / stem)
    save_json(inputs / "probe_source.json", dict(directory=str(pilot),
              source_manifest_sha256=digest(pilot / "source_sha256.json")))
    save_json(inputs / "models.json", models)
    for name in ["target.vtp", "target.json", "target_outline_m.npy"]:
        shutil.copy2(position.TARGET / name, folder / name)
    shutil.copytree(ROBOT, folder / "franka/panda_model_snapshot")
    save_json(folder / "protocol.json", PROTOCOL)
    paths = [*ROOT.joinpath("src").rglob("*.py"),
             *ROOT.joinpath("experiments/robotics").glob("*.py"),
             ROOT / "experiments/nclaw/suite.py"]
    hashes = {}
    for path in paths:
        if path.name == "x_common_report.py":
            continue  # The renderer has its own source/provenance archive.
        rel = path.relative_to(ROOT)
        dest = folder / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[str(rel)] = digest(path)
    save_json(folder / "source_sha256.json", hashes)
    save_json(folder / "input_sha256.json", {str(p.relative_to(folder)): digest(p)
              for p in [*inputs.iterdir(), *folder.glob("target*")] if p.is_file()})
    save_json(folder / "robot_input_sha256.json", {str(p.relative_to(folder)): digest(p)
              for p in (folder / "franka/panda_model_snapshot").rglob("*") if p.is_file()})
    for rel in ["paper.tex", "paper.pdf", "figs/identification_plastic_shaping.pdf", "figs/identification_plastic_shaping.png"]:
        dest = folder / "manuscript_before" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "paper/icra2027" / rel, dest)
    (folder / "packages.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
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


# Reuse the tested search, controller and checks without inheriting old fits/plans.
position.check = work.check = check


def validate(folder, name, device):
    check(folder)
    models = json.loads((folder / "inputs/models.json").read_text())
    for model in [f"true_{name}", f"identified_{name}"]:
        path = folder / "validation" / f"{model}.npz"
        assert not path.exists(), "Validation already recorded"
        data, phases = probe.execute(models[model], protocol=probe.VALIDATION, device=device, states=False)
        save_npz(path, data)
        save_json(path.with_suffix(".phases.json"), phases)
        save_json(path.with_suffix(".config.json"), dict(law=models[model], protocol=probe.VALIDATION,
                  device=device, n_grid=64, dt=.00005, protocol_sha256=digest(folder / "protocol.json")))
        print(json.dumps(dict(validation=model, data_sha256=digest(path))), flush=True)


def predict(folder, name, device):
    check(folder)
    selection = json.loads((folder / f"plans/identified_{name}/selected.json").read_text())
    path = folder / f"prediction_{name}.npz"
    data, _ = work.rollout(folder, path, selection["law"], gaps=selection["gaps_mm"], device=device)
    old = np.load(folder / selection["file"])
    rms = float(np.sqrt(np.mean(np.sum((data["x_after_1s"] - old["x_after_1s"]) ** 2, axis=-1))) * 1000)
    assert rms < .5
    save_json(folder / f"plans/{name}/selected.json", dict(model=f"identified_{name}",
        law=selection["law"], budgets_j=data["actual_work_j"].tolist(),
        gaps_mm=selection["gaps_mm"], search_evaluations=selection["evaluations"],
        position_selection_sha256=digest(folder / f"plans/identified_{name}/selected.json"),
        fixed_speed_replay_rms_difference_mm=rms, surface_mm=position.score(data, pv.read(folder / "target.vtp")),
        file=path.name, data_sha256=digest(path)))


def verify_identification(folder):
    check(folder)
    a, b = [np.load(folder / f"inputs/probe_{n}.npz") for n in "AB"]
    for key in ["initial", "vol0", "time", "tool_centers", "tool_velocity", "phase_id", "state_time"]:
        np.testing.assert_array_equal(a[key], b[key])
    models = json.loads((folder / "inputs/models.json").read_text())
    errors = {}
    for n in "AB":
        true = np.load(folder / f"validation/true_{n}.npz")
        active = np.isin(true["phase_id"], [2, 3, 4])
        force = true["reaction_force"][active]
        errors[n] = {}
        for planned in "AB":
            predicted = np.load(folder / f"validation/identified_{planned}.npz")
            np.testing.assert_array_equal(true["tool_centers"], predicted["tool_centers"])
            errors[n][planned] = float(100 * np.linalg.norm(predicted["reaction_force"][active] - force) / np.linalg.norm(force))
        fit = json.loads((folder / f"inputs/identification_{n}.json").read_text())
        assert fit["law"] == models[f"identified_{n}"]
        selection = json.loads((folder / f"plans/identified_{n}/selected.json").read_text())
        assert selection["law"] == fit["law"] and selection["evaluations"] == 32
        rows = json.loads((folder / f"plans/identified_{n}/evaluations.json").read_text())
        assert selection["file"] == min(rows, key=lambda r: r["surface_mm"])["file"]
    save_json(folder / "identification_validation.json", dict(exact_same_probe_trajectory=True,
              force_relative_l2_percent=errors, force_metric="Both fingers' 3D reaction vectors over close/hold/open; no time alignment or force fitting",
              fresh_specimens=True))
    print(json.dumps(errors), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "validate", "plan", "predict", "selfcheck", "evaluate", "verify"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--pilot", type=Path, default=PILOT)
    p.add_argument("--material", choices=["A", "B"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--settings", choices=list(position.SETTINGS), nargs="+", default=list(position.SETTINGS))
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.out, args.pilot)
    elif args.stage == "validate":
        validate(args.out, args.material, args.device)
    elif args.stage == "plan":
        position.plan(args.out, f"identified_{args.material}", args.device)
    elif args.stage == "predict":
        predict(args.out, args.material, args.device)
    elif args.stage == "selfcheck":
        work.selfcheck(args.out, args.material, args.device)
    elif args.stage == "evaluate":
        work.evaluate(args.out, args.material, args.device, args.settings)
    else:
        verify_identification(args.out)
        work.verify(args.out)


if __name__ == "__main__":
    main()
