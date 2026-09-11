"""Plan smooth X-shaping motions and retain their forces without retiming.

The same simulator and motion generator serve every planning candidate and
final execution. The selected candidate's stored force and velocity arrays
become the execution references directly. Historical studies stay frozen.
"""
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

from experiments.robotics import x_force_profile_control as control
from experiments.robotics import x_force_profile_study as prior
from experiments.robotics import x_force_profile_gains as execution
from experiments.robotics.plastic_shaping_study import save_json, save_npz
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.hex_shaping_surface import mesh_distance_mm

ROOT = prior.ROOT
SOURCE = ROOT / "out/x_force_profile_study_20260910"
OUT = ROOT / "out/x_consistent_force_study_20260911"
SETTINGS = prior.SETTINGS
digest = prior.digest
PROTOCOL = dict(
    scene=prior.core.CONFIG, control=control.CONTROL, settings=SETTINGS,
    seed=0, reference_grid=64, reference_dt=.00005,
    initial_gaps_mm=[20., 31., 35., 20.], simplex_offset_mm=-3.,
    bounds_mm=[[16., 54.]] * 4, max_evaluations=32,
    search="Bounded Nelder-Mead in each identified model; best observed surface error within the fixed evaluation budget",
    planning="Every candidate uses smooth 40 ms ramps and <=20 mm/s per finger; prescribed motion, force feedback disabled",
    reference="Copy the selected planning rollout byte-for-byte; extract its saved force and baseline velocity arrays without another simulation or retiming",
    objective="Symmetric area-weighted mean 3D surface distance to the unchanged rounded X, 1 s after final withdrawal; 1.25 mm voxels, Gaussian sigma 1.625 mm, isovalue 0.5",
    execution="Unchanged PI force feedback at 4 ms, plus the selected baseline velocity; profile duration or 12 mm guard stops; no final-gap enforcement or shape feedback",
    controller="Keep the previous common gain multiplier 0.5; check both identified models at baseline and finer grid, plus baseline repeats before true execution; stop if these checks fail",
    feasibility="All four profiles complete; <=15% raw-force relative L2 error per profile; sampled height <=45.5 mm",
    comparison="Each entire frozen plan executes in both true materials at three numerical settings",
    initialization="Same common initialization and target from the previous studies, originally informed by A feasibility trials; no material-specific warm starts",
    scope="Supplied-state identification and ideal simulated finger-force sensing; kinematic tools; hardware force control and numerical convergence are not established",
)
GAIN_PROTOCOL = dict(
    gain_multiplier=.5, selection="Fixed from the prior identified-model controller study; no new gain search",
    source_archive="out/x_force_profile_study_20260910",
    validation="Both identified models at 64^3/50us and 80^3/50us plus a baseline repeat before freezing",
)


def prepare(folder, source=SOURCE):
    execution.check(source)
    for rel, sha in json.loads((source / "checksums.json").read_text()).items():
        assert digest(source / rel) == sha, rel
    assert not folder.exists(), "Use a fresh study directory"
    folder.mkdir(parents=True)
    for name in ["inputs", "validation"]:
        shutil.copytree(source / name, folder / name)
    shutil.copytree(source / "franka/panda_model_snapshot", folder / "franka/panda_model_snapshot")
    for name in ["target.vtp", "target.json", "target_outline_m.npy",
                 "identification_validation.json", "observation_isolation.json"]:
        shutil.copy2(source / name, folder / name)
    models = json.loads((folder / "inputs/models.json").read_text())
    for material in "AB":
        save_json(folder / f"inputs/initialization_{material}.json", dict(
            model=f"identified_{material}", law=models[f"identified_{material}"],
            gaps_mm=PROTOCOL["initial_gaps_mm"], scope="Common search initialization; not the selected plan"))
    save_json(folder / "protocol.json", PROTOCOL)
    save_json(folder / "gain_protocol.json", GAIN_PROTOCOL)
    save_json(folder / "input_provenance.json", dict(
        source_archive=str(source.resolve()), source_manifest_sha256=digest(source / "checksums.json"),
        identification="Copied frozen supplied-state fits and held-out validation; not re-estimated",
        target="Copied unchanged", planning="Fresh 32-evaluation search per material with consistent motion timing"))
    sources = set(json.loads((source / "source_sha256.json").read_text()))
    sources.update("experiments/robotics/" + name for name in [
        "x_consistent_force_study.py", "x_force_profile_gains.py", "x_force_profile_audit.py"])
    hashes = {}
    for rel in sorted(sources):
        dst = folder / "source_snapshot" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
        hashes[rel] = digest(dst)
    save_json(folder / "source_sha256.json", hashes)
    save_json(folder / "input_sha256.json", {str(p.relative_to(folder)): digest(p)
        for p in [*folder.joinpath("inputs").iterdir(), *folder.joinpath("validation").iterdir(),
                  *folder.glob("target*"), folder / "identification_validation.json", folder / "observation_isolation.json"]
        if p.is_file()})
    save_json(folder / "robot_input_sha256.json", {str(p.relative_to(folder)): digest(p)
        for p in folder.joinpath("franka/panda_model_snapshot").rglob("*") if p.is_file()})
    (folder / "packages.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    (folder / "commit.txt").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"]))
    (folder / "tracked_before.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"]))
    (folder / "gpu_before.txt").write_bytes(subprocess.check_output(["nvidia-smi"]))
    for name in ["paper.tex", "paper.pdf", "paper.log", "paper.blg",
                 "figs/identification_plastic_shaping.pdf", "figs/identification_plastic_shaping.png"]:
        src = ROOT / "paper/icra2027" / name
        if src.exists():
            dst = folder / "manuscript_before" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    check(folder)


def check(folder):
    assert json.loads((folder / "protocol.json").read_text()) == json.loads(json.dumps(PROTOCOL))
    assert json.loads((folder / "gain_protocol.json").read_text()) == GAIN_PROTOCOL
    for rel, sha in json.loads((folder / "source_sha256.json").read_text()).items():
        assert digest(ROOT / rel) == digest(folder / "source_snapshot" / rel) == sha, rel
    for name in ["input_sha256.json", "robot_input_sha256.json"]:
        for rel, sha in json.loads((folder / name).read_text()).items():
            assert digest(folder / rel) == sha, rel


def plan(folder, material, device):
    check(folder)
    assert not (folder / "execution_plan.json").exists(), "Plans already frozen"
    model = f"identified_{material}"
    law = json.loads((folder / "inputs/models.json").read_text())[model]
    target = pv.read(folder / "target.vtp")
    dest = folder / "plans" / model
    dest.mkdir(parents=True, exist_ok=True)
    rows = []

    def objective(gaps):
        gaps = np.round(gaps, 6)
        key = hashlib.sha256(gaps.tobytes()).hexdigest()[:16]
        path = dest / f"{key}.npz"
        cfg = dict(model=model, law=law, gaps_mm=gaps.tolist(), device=device,
                   n_grid=64, dt=.00005, seed=0, reference_generation=True,
                   protocol_sha256=digest(folder / "protocol.json"))
        start = time.monotonic()
        cached = path.exists() and path.with_suffix(".json").exists()
        if cached:
            assert json.loads(path.with_suffix(".config.json").read_text()) == cfg
            record = json.loads(path.with_suffix(".json").read_text())
            assert record["data_sha256"] == digest(path)
        else:
            save_json(path.with_suffix(".config.json"), cfg)
            profiles = control.nominal_profiles(gaps)
            data, phases, stops = control.execute(law, profiles, 64, .00005, device, reference=True)
            assert stops == ["profile_complete"] * 4
            np.testing.assert_allclose(data["min_gaps_mm"], gaps, atol=1e-6)
            assert np.max(np.abs(data["force_error"])) == 0
            save_npz(path, data)
            save_json(path.with_suffix(".phases.json"), phases)
            error = mesh_distance_mm(surface(data["x_after_1s"], data["vol0"], h=.00125), target)
            record = dict(**cfg, file=str(path.relative_to(folder)), surface_mm=error,
                          data_sha256=digest(path), elapsed_s=time.monotonic() - start)
            save_json(path.with_suffix(".json"), record)
        rows.append(dict(**record, reused_cached_result=cached))
        save_json(dest / "evaluations.json", rows)
        print(json.dumps(dict(material=material, evaluation=len(rows), gaps_mm=gaps.tolist(),
                              surface_mm=record["surface_mm"], cached=cached,
                              elapsed_s=time.monotonic() - start)), flush=True)
        return record["surface_mm"]

    initial = np.array(PROTOCOL["initial_gaps_mm"])
    simplex = np.tile(initial, (5, 1))
    simplex[1:] += PROTOCOL["simplex_offset_mm"] * np.eye(4)
    result = minimize(objective, initial, method="Nelder-Mead", bounds=PROTOCOL["bounds_mm"],
                      options=dict(maxfev=PROTOCOL["max_evaluations"], xatol=0., fatol=0., initial_simplex=simplex))
    best = min(rows, key=lambda row: row["surface_mm"])
    selected = dict(**best, evaluations=len(rows), unique_simulations=len({r["file"] for r in rows}),
                    optimizer_success=bool(result.success), optimizer_message=str(result.message))
    save_json(dest / "selected.json", selected)
    selected_path = folder / best["file"]
    stem = folder / f"reference_{material}"
    for suffix in [".npz", ".phases.json", ".config.json"]:
        shutil.copy2(selected_path.with_suffix(suffix), stem.with_suffix(suffix))
    assert digest(stem.with_suffix(".npz")) == best["data_sha256"]
    with np.load(selected_path) as data:
        profiles = control.reference_profiles(data)
    prior.save_profiles(folder / f"profiles_{material}.npz", profiles)
    save_json(stem.with_suffix(".json"), dict(**{k: v for k, v in best.items() if k != "file"},
        file=stem.with_suffix(".npz").name, selected_planning_file=best["file"],
        selected_planning_sha256=best["data_sha256"],
        profiles_sha256=digest(folder / f"profiles_{material}.npz"),
        peaks_n=[float(p["peak_n"]) for p in profiles],
        durations_s=[len(p["velocity_ff"]) * .004 for p in profiles],
        additional_reference_simulations=0, retimed=False))
    print(json.dumps(dict(selected=material, **selected)), flush=True)


def model_checks(folder, material, device):
    check(folder)
    assert not (folder / "execution_plan.json").exists()
    law = json.loads((folder / "inputs/models.json").read_text())[f"identified_{material}"]
    rows = []
    baseline = None
    for setting, (grid, dt) in {"baseline": (64, .00005), "grid80": (80, .00005), "repeat_baseline": (64, .00005)}.items():
        path = folder / f"model_checks/{material}/{setting}.npz"
        data, record = execution.rollout(folder, path, law, material, device, grid, dt, .5)
        rows.append(dict(**record, setting=setting))
        save_json(folder / f"model_checks/{material}/results.json", rows)
        if setting == "baseline":
            baseline = (np.asarray(data["x_after_1s"]).copy(), record)
        if setting == "repeat_baseline":
            repeat = dict(surface_error_difference_mm=abs(baseline[1]["surface_mm"] - record["surface_mm"]),
                particle_rms_difference_mm=float(np.sqrt(np.mean(np.sum((baseline[0] - data["x_after_1s"]) ** 2, axis=1)))) * 1000,
                same_stops=baseline[1]["stops"] == record["stops"])
            save_json(folder / f"model_checks/{material}/repeat_check.json", repeat)
    print(json.dumps(dict(material=material, all_feasible=all(r["feasible"] for r in rows), repeat=repeat)), flush=True)


def freeze(folder):
    check(folder)
    assert not (folder / "execution_plan.json").exists()
    plans = {}
    for material in "AB":
        rows = json.loads((folder / f"model_checks/{material}/results.json").read_text())
        repeat = json.loads((folder / f"model_checks/{material}/repeat_check.json").read_text())
        assert len(rows) == 3 and all(r["feasible"] for r in rows), material
        assert repeat["same_stops"] and repeat["surface_error_difference_mm"] < .1 and repeat["particle_rms_difference_mm"] < .5
        plans[material] = dict(profiles_sha256=digest(folder / f"profiles_{material}.npz"),
            reference_sha256=digest(folder / f"reference_{material}.json"), model_checks=rows, repeat=repeat,
            selection_sha256=digest(folder / f"plans/identified_{material}/selected.json"))
    save_json(folder / "gain_selected.json", GAIN_PROTOCOL)
    save_json(folder / "execution_plan.json", dict(plans=plans, gain_multiplier=.5,
        gain_selected_sha256=digest(folder / "gain_selected.json"), protocol_sha256=digest(folder / "protocol.json"),
        gain_protocol_sha256=digest(folder / "gain_protocol.json"),
        scope="Selected planning trajectories, their recorded forces, and unchanged common controller frozen before true-material execution"))


def evaluate(folder, material, device, settings):
    check(folder)
    frozen = json.loads((folder / "execution_plan.json").read_text())
    assert frozen["gain_selected_sha256"] == digest(folder / "gain_selected.json")
    laws = json.loads((folder / "inputs/models.json").read_text())
    for planned in "AB":
        assert frozen["plans"][planned]["profiles_sha256"] == digest(folder / f"profiles_{planned}.npz")
    for setting in settings:
        grid, dt = SETTINGS[setting]
        for planned in "AB":
            path = folder / f"{setting}_{material}_plan_{planned}.npz"
            _, record = execution.rollout(folder, path, laws[f"true_{material}"], planned, device, grid, dt, .5)
            save_json(path.with_suffix(".json"), dict(**record, material=material, setting=setting))


def audit(folder):
    from experiments.robotics import x_force_profile_audit as verifier
    check(folder)
    rows = []
    for material in "AB":
        selected = json.loads((folder / f"plans/identified_{material}/selected.json").read_text())
        reference = json.loads((folder / f"reference_{material}.json").read_text())
        assert selected["evaluations"] == PROTOCOL["max_evaluations"]
        assert digest(folder / selected["file"]) == digest(folder / f"reference_{material}.npz") == selected["data_sha256"]
        assert not reference["retimed"] and reference["additional_reference_simulations"] == 0
        profiles = prior.load_profiles(folder / f"profiles_{material}.npz")
        generated = control.nominal_profiles(selected["gaps_mm"])
        with np.load(folder / selected["file"]) as d:
            np.testing.assert_allclose(d["min_gaps_mm"], selected["gaps_mm"], atol=1e-6)
            for i, (p, nominal) in enumerate(zip(profiles, generated, strict=True)):
                np.testing.assert_array_equal(p["velocity_ff"], nominal["velocity_ff"])
                mask = d["pulse_id"] == i
                np.testing.assert_array_equal(p["force_reference"], d["force_per_finger"][mask].mean(axis=1))
                np.testing.assert_array_equal(p["feedback_reference"], d["filtered_force_before"][mask])
            assert not d["force_error"].any()
        rows.append(dict(material=material, identical_selected_rollout_and_reference=True,
            identical_planning_and_execution_baseline_velocity=True, forces_from_selected_rollout=True,
            additional_reference_simulations=0, evaluations=selected["evaluations"],
            unique_simulations=selected["unique_simulations"]))
    save_json(folder / "planning_consistency_audit.json", rows)
    verifier.audit(folder, validate=check)
    verifier.diagnostics(folder)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "plan", "model_checks", "freeze", "evaluate", "audit", "report"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--material", choices=["A", "B"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    p.add_argument("--render-out", type=Path)
    a = p.parse_args()
    if a.stage == "prepare": prepare(a.out, a.source)
    elif a.stage == "evaluate": evaluate(a.out, a.material, a.device, a.settings)
    elif a.stage in ["plan", "model_checks"]: globals()[a.stage](a.out, a.material, a.device)
    elif a.stage == "report":
        from experiments.robotics.x_force_profile_report import report
        assert a.render_out is not None
        report(a.out, a.render_out, validate=check)
    else: globals()[a.stage](a.out)
