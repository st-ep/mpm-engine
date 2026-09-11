"""Execute the frozen two-gap plans with a one-second raised quarter turn.

The original planning results remain unchanged. This checks a slower noncontact
reorientation for the proposed Panda execution; it introduces no new controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.robotics.x_shaping import CONFIG, execute, score
from experiments.robotics.x_shaping_study import OUT, MODELS, SETTINGS, check_sources
from experiments.robotics.plastic_shaping_study import save_json, save_npz, chamfer_mm


def prepare(source):
    check_sources(source); folder = source/"hardware_timing"; folder.mkdir(exist_ok=True)
    protocol = dict(planner_source=str(source.resolve()),
                    planner_protocol_sha256=hashlib.sha256((source/"protocol.json").read_bytes()).hexdigest(),
                    change="Raised 90-degree turn takes 1.0 s instead of 0.5 s; selected gaps and other motion phases unchanged",
                    rotation_speed_rad_s=np.pi/2, scene=dict(CONFIG, rotation_speed=np.pi/2),
                    observations="1 s after final withdrawal", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = folder/"execution_protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        save_json(path, protocol); shutil.copy2(__file__, folder/Path(__file__).name)
    return folder


def run(source, models, device):
    folder = prepare(source); laws = json.loads((source/"inputs/models.json").read_text())
    for model in models:
        check_sources(source)
        selected = json.loads((source/"plans"/model/"selected.json").read_text())
        names = "AB" if model == "nominal" else model[-1]
        method = "nominal" if model == "nominal" else ("oracle" if model.startswith("true_") else "identified")
        for setting, (grid, dt) in SETTINGS.items():
            for name in names:
                path = folder/f"{setting}_{name}_{method}.npz"; start = time.monotonic()
                if not path.exists():
                    original_speed = CONFIG["rotation_speed"]
                    try:
                        CONFIG["rotation_speed"] = np.pi/2
                        data, phases = execute(laws[f"true_{name}"], selected["gaps_mm"], device=device, n_grid=grid, dt=dt)
                    finally:
                        CONFIG["rotation_speed"] = original_speed
                    save_npz(path, data); save_json(path.with_suffix(".phases.json"), phases)
                data = np.load(path); original = np.load(source/path.name)
                record = dict(material=name, model=model, method=method, setting=setting, gaps_mm=selected["gaps_mm"],
                              surface_mm=score(data), original_surface_mm=score(original),
                              retiming_cloud_difference_mm=chamfer_mm(data["x_after_1s"], original["x_after_1s"]),
                              peak_force_per_finger_n=np.linalg.norm(data["reaction_force"], axis=-1).max(axis=0).tolist(),
                              rms_speed_m_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"]**2, axis=1)))),
                              original_file_sha256=hashlib.sha256((source/path.name).read_bytes()).hexdigest())
                save_json(path.with_suffix(".json"), record)
                print(json.dumps(dict(seconds=time.monotonic()-start, **record)), flush=True)


def verify(source):
    folder = prepare(source); results = []
    for path in sorted(folder.glob("*.npz")):
        d = np.load(path); r = json.loads(path.with_suffix(".json").read_text())
        assert int(d["inverted_count"]) == 0 and all(np.all(np.isfinite(d[k])) for k in d.files)
        np.testing.assert_allclose(score(d), r["surface_mm"], rtol=1e-10)
        selected = json.loads((source/"plans"/r["model"]/"selected.json").read_text())
        np.testing.assert_array_equal(d["gaps_mm"], selected["gaps_mm"])
        phases = json.loads(path.with_suffix(".phases.json").read_text())
        old = json.loads((source/path.with_suffix(".phases.json").name).read_text())
        assert len(phases) == len(old) == 10
        for i, (p, before) in enumerate(zip(phases, old, strict=True)):
            assert p["name"] == before["name"]
            np.testing.assert_array_equal(p["start_centers"], before["start_centers"])
            np.testing.assert_array_equal(p["end_centers"], before["end_centers"])
            np.testing.assert_allclose(p["duration_s"], 1. if p["name"].endswith("rotate") else before["duration_s"])
            poses = d["tool_centers"][d["phase_id"] == i]
            np.testing.assert_allclose(poses[-1], p["end_centers"], atol=1e-10)
        assert hashlib.sha256((source/path.name).read_bytes()).hexdigest() == r["original_file_sha256"]
        results.append(r)
    assert len(results) == 18
    save_json(folder/"summary.json", dict(rollouts=len(results), results=results))
    print(json.dumps(dict(rollouts=len(results), maximum_cloud_change_mm=max(r["retiming_cloud_difference_mm"] for r in results))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("stage", choices=["run", "verify"])
    p.add_argument("--out", type=Path, default=OUT); p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.stage == "run":
        run(args.out.resolve(), args.models, args.device)
    else:
        verify(args.out.resolve())
