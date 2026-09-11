"""Reproduce the separate supplied-state calibration for the cylindrical study.

Both materials use E=80 kPa, nu=.30; yield thresholds are 1 and 10 kPa.
These are the earlier rectangular study's material definitions, not the later
hexagonal study's 240 kPa material B. No shaping outcomes enter these fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.robotics.plastic_shaping_study import ROOT, TRUTH, NOMINAL, press, identify, save_json, save_npz

OUT = ROOT/"out/x_shaping_identification_20260909"


def prepare(folder, device):
    folder.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT/"experiments/robotics/plastic_shaping_study.py",
               ROOT/"experiments/nclaw/suite.py", *sorted((ROOT/"src/warpmpm").rglob("*.py")),
               *sorted((ROOT/"src/ident").rglob("*.py")), *sorted((ROOT/"src/common").rglob("*.py"))]
    hashes = {}
    for path in sources:
        rel = str(path.relative_to(ROOT)); dest = folder/"source_snapshot"/rel
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if dest.exists():
            assert hashlib.sha256(dest.read_bytes()).hexdigest() == h
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest); hashes[rel] = h
    save_json(folder/"source_sha256.json", hashes)
    save_json(folder/"protocol.json", dict(truth=TRUTH, nominal=NOMINAL, press_depth_m=.014,
              press_speed_m_s=.08, validation_depth_m=.024, validation_speed_m_s=.12,
              observations=["noise-free x", "v", "elastic F", "mass", "reference volume"],
              known=["geometry", "density", "contact", "Hencky/von-Mises constitutive family"],
              scope="Simulation identification only; no RGB-D or hardware identification claim"))
    models = {"nominal": NOMINAL, **{f"true_{n}": law for n, law in TRUTH.items()}}
    for name, law in TRUTH.items():
        path = folder/f"probe_{name}.npz"
        if not path.exists():
            save_npz(path, press(law, device=device))
        fit = identify(dict(np.load(path))); models[f"identified_{name}"] = fit["law"]
        save_json(folder/f"identification_{name}.json", fit)
        print(json.dumps(dict(material=name, **fit)), flush=True)
    save_json(folder/"models.json", models)
    errors = {}
    for model, law in models.items():
        path = folder/f"validation_{model}.npz"
        if not path.exists():
            save_npz(path, press(law, device=device, depth=.024, speed=.12, record_state=False))
    for name in TRUTH:
        true = np.load(folder/f"validation_true_{name}.npz")["force"]
        errors[name] = {}
        for method, model in [("nominal", "nominal"), ("identified", f"identified_{name}")]:
            predicted = np.load(folder/f"validation_{model}.npz")["force"]
            errors[name][method] = float(np.linalg.norm(predicted-true)/np.linalg.norm(true)*100)
    save_json(folder/"force_errors_percent.json", errors); print(json.dumps(errors), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); prepare(args.out.resolve(), args.device)
