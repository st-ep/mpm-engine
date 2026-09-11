"""Reproduce smooth-motion planning, force-feedback execution, audit and figure.

Identification and target construction are imported frozen inputs. Both
32-evaluation searches are rerun; force references come directly from their
selected rollouts. This command does not modify the active paper.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.robotics import x_consistent_force_study as study
from experiments.robotics.plastic_shaping_study import save_json


def reproduce(source, folder, render, devices, prepare_only=False):
    study.check(source)
    manifest = json.loads((source / "checksums.json").read_text())
    for rel, sha in manifest.items():
        assert study.digest(source / rel) == sha, rel
    assert not folder.exists() and not render.exists(), "Use fresh output directories"
    folder.mkdir(parents=True)
    for name in ["inputs", "validation", "source_snapshot"]:
        shutil.copytree(source / name, folder / name)
    shutil.copytree(source / "franka/panda_model_snapshot", folder / "franka/panda_model_snapshot")
    for name in ["protocol.json", "gain_protocol.json", "source_sha256.json", "input_sha256.json",
                 "robot_input_sha256.json", "target.vtp", "target.json", "target_outline_m.npy",
                 "identification_validation.json", "observation_isolation.json", "packages.txt", "commit.txt"]:
        shutil.copy2(source / name, folder / name)
    shutil.copy2(__file__, folder / "reproduction_source.py")
    (folder / "replay_packages.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    save_json(folder / "reproduction_provenance.json", dict(source_archive=str(source.resolve()),
        source_manifest_sha256=study.digest(source / "checksums.json"), devices=devices,
        scope="Frozen identification and target inputs; new searches, extracted references, model checks, true executions, audit and figure"))
    study.check(folder)
    save_json(folder / "reproduction_staging_audit.json", dict(
        source_and_input_hashes_checked=True, prior_plans_or_executions_imported=False))
    if prepare_only:
        print(json.dumps(dict(prepared=str(folder), simulations_run=0)), flush=True)
        return
    logs = folder / "logs"
    logs.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def run(stage, material=None, device=None):
        cmd = [sys.executable, "-m", "experiments.robotics.x_consistent_force_study",
               stage, "--out", str(folder)]
        if material:
            cmd += ["--material", material, "--device", device]
        if stage == "report":
            cmd += ["--render-out", str(render)]
        log = logs / f"{stage}{'_' + material if material else ''}.log"
        print(json.dumps(dict(command=cmd, log=str(log))), flush=True)
        with log.open("w") as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)

    def model(material, device):
        run("plan", material, device)
        run("model_checks", material, device)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(model, m, d) for m, d in zip("AB", devices, strict=True)]
        for future in futures:
            future.result()
    run("freeze")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, "evaluate", m, d) for m, d in zip("AB", devices, strict=True)]
        for future in futures:
            future.result()
    run("audit")
    run("report")
    comparisons = []
    for setting in study.SETTINGS:
        for material in "AB":
            for planned in "AB":
                name = f"{setting}_{material}_plan_{planned}.json"
                original = json.loads((source / name).read_text())
                repeated = json.loads((folder / name).read_text())
                comparisons.append(dict(case=name,
                    surface_error_difference_mm=abs(original["surface_mm"] - repeated["surface_mm"]),
                    same_stops=original["stops"] == repeated["stops"]))
    save_json(folder / "reproduction_comparison.json", dict(results=comparisons,
        scope="GPU reductions and optimizer paths need not be bit-identical; report differences without overwriting original results"))
    save_json(folder / "checksums.json", {str(p.relative_to(folder)): study.digest(p)
        for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "checksums.json"})
    print(json.dumps(dict(completed=str(folder), figure=str(render / "identification_plastic_shaping.png"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=study.OUT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-out", type=Path, required=True)
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    reproduce(args.source, args.out, args.render_out, args.devices, args.prepare_only)
