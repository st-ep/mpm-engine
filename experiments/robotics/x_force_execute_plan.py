"""Execute one frozen force plan as soon as its independent search finishes.

This schedules the same six true-material/numerical-condition executions as
x_force_study.evaluate, grouped by planning model to keep both GPUs occupied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import pyvista as pv

from experiments.robotics.x_force_study import OUT, SETTINGS, check, digest, rollout, score
from experiments.robotics.x_force_control import diagnostics
from experiments.robotics.plastic_shaping_study import save_json


def execute_plan(folder, planned_for, device):
    check(folder)
    selected = json.loads((folder / f"plans/{planned_for}/selected.json").read_text())
    target = pv.read(folder / "target.vtp")
    laws = json.loads((folder / "inputs/models.json").read_text())
    source = folder / f"execution_plan_{planned_for}_source.py"
    if source.exists():
        assert digest(source) == digest(Path(__file__))
    else:
        shutil.copy2(__file__, source)
    save_json(folder / f"execution_plan_{planned_for}.json", dict(
        planned_for=planned_for, device=device, source_sha256=digest(source),
        selected_sha256=digest(folder / f"plans/{planned_for}/selected.json"),
        scope="Identical physics, commands, and scoring; cases grouped by planning model for scheduling"))
    for setting, (grid, dt) in SETTINGS.items():
        for material in "AB":
            path = folder / f"{setting}_{material}_plan_{planned_for}.npz"
            print(json.dumps(dict(starting=path.name)), flush=True)
            start = time.monotonic()
            data, phases = rollout(folder, path, laws[f"true_{material}"], selected["peaks_n"], grid, dt, device)
            record = dict(material=material, planned_for=planned_for, setting=setting,
                peaks_n=selected["peaks_n"], surface_mm=score(data, target),
                diagnostics=diagnostics(data, phases), file=path.name, data_sha256=digest(path))
            save_json(path.with_suffix(".json"), record)
            print(json.dumps(dict(**record, elapsed_s=time.monotonic() - start)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--plan", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    execute_plan(args.out, args.plan, args.device)
