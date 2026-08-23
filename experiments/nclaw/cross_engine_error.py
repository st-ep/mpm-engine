"""Engine-gap decomposition on NCLaw-generated trajectories.

For one material, given ingested truth dumps under
mpm_engine/out/nclaw_cross_generalize/dumps/<material>_<scene>_truth.npz:

  1. identify theta from the DATASET scene alone (the protocol's rule);
  2. per scene, roll our engine from THEIR frame-0 cloud at (a) the truth
     parameters and (b) the recovered parameters, at their own frame cadence;
  3. score both against their trajectory in their metric.

Row (a) is the error at the true parameters: no identification method can score below it. The ratio
(b)/(a) is the identification excess, the only part of a cross-engine cell
that identification quality controls.

--nclaw-bc turns on the engine's NCLaw-compatibility grid mode (their
approach-only wall clamp, eps-softened mass division, free-fall velocity on
empty nodes, MLS transfer and their particle position clamp), which is what the
floor measures the value of. --nclaw-law additionally rolls the material out on
NCLaw's own constitutive pair for it (engine material 14) and identifies in that
form, which matters wherever their law is not a reparameterization of ours:
their water is a linear volumetric EOS with no deviatoric term at all.
--substeps=N fixes the substep count per dumped frame; --substeps=1 is what
matches their integrator exactly, and their sand needs it because our CFL picks
two substeps where they take one. --bisect runs the dataset scene's truth-theta
leg with one behavior at a time so the floor's drop is attributable.

Run:  .venv/bin/python -m experiments.nclaw.cross_engine_error plasticine \
          [--nclaw-bc] [--nclaw-law] [--substeps=1] [--bisect]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DUMPS = ROOT / "out" / "nclaw_cross_generalize" / "dumps"
OUT = ROOT / "out" / "nclaw_cross_floor"

# One behavior at a time, against the engine's default grid path. "off" is the
# baseline (no mode at all); "full" is every behavior at their dataset settings.
OFF = {"freeslip_bound": 0, "mass_eps": 0.0, "empty_node_gravity": False,
       "mls_transfer": False, "particle_clip_cells": -1.0}
BISECT_LEGS = {
    "off": False,
    "freeslip_walls": {**OFF, "freeslip_bound": 3},
    "mass_eps": {**OFF, "mass_eps": 1.0e-7},
    "empty_node_gravity": {**OFF, "empty_node_gravity": True},
    "mls_transfer": {**OFF, "mls_transfer": True},
    "particle_clip": {**OFF, "particle_clip_cells": 0.5},
    "full": True,
    # leave-one-out, to attribute the residual the walls alone leave behind
    "full_no_mls": {"mls_transfer": False},
    "full_no_mass_eps": {"mass_eps": 0.0},
    "full_no_empty_gravity": {"empty_node_gravity": False},
    "full_no_clip": {"particle_clip_cells": -1.0},
    "walls_and_mls": {**OFF, "freeslip_bound": 3, "mls_transfer": True},
}


def main(material: str, nclaw_bc: bool = False, bisect: bool = False,
         nclaw_law: bool = False, substeps: int | None = None) -> None:
    from experiments.nclaw.suite import (
        MATERIALS,
        cloud_from_dump,
        nclaw_position_mse,
        run_scene,
        stage_identify,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = sorted(p.name.split(f"{material}_", 1)[1].rsplit("_truth.npz", 1)[0]
                    for p in DUMPS.glob(f"{material}_*_truth.npz"))
    if "dataset" not in scenes:
        raise SystemExit(f"no {material}_dataset_truth.npz under {DUMPS}")
    theta_true = dict(MATERIALS[material]["truth"])

    if bisect:
        # attribution run: the dataset scene at truth theta, one behavior at a
        # time. The grid mode changes no identification input, so theta is fixed.
        truth = DUMPS / f"{material}_dataset_truth.npz"
        cloud = cloud_from_dump(truth)
        rows = {}
        for leg, mode in BISECT_LEGS.items():
            pred = OUT / f"{material}_dataset_truth_theta_bis_{leg}.npz"
            if not pred.exists():
                t0 = time.time()
                run_scene(material, "dataset", pred, theta=dict(theta_true),
                          cloud=cloud, nclaw_bc=mode, nclaw_law=nclaw_law,
                          substeps=substeps)
                print(f"[bisect] {leg} simulated in {time.time() - t0:.0f}s")
            s = nclaw_position_mse(truth, pred)
            rows[leg] = {k: s[k] for k in
                         ("mse", "mse_final_frame", "rmse_mm", "n_frames")}
            print(f"[bisect] {leg}: MSE {s['mse']:.4e} "
                  f"(RMS {s['rmse_mm']:.2f} mm)")
        base = rows["off"]["mse"]
        for leg, cell in rows.items():
            cell["ratio_to_off"] = cell["mse"] / base if base > 0 else None
        path = OUT / f"bisect_{material}.json"
        path.write_text(json.dumps(
            {"material": material, "scene": "dataset", "theta": theta_true,
             "legs": BISECT_LEGS, "rows": rows}, indent=2, default=float))
        print(f"[bisect] wrote {path}")
        return

    tag = (("_nclawbc" if nclaw_bc else "") + ("_nclawlaw" if nclaw_law else "")
           + (f"_sub{substeps}" if substeps is not None else ""))
    print(f"[floor] {material}: scenes {scenes} nclaw_bc={nclaw_bc} "
          f"nclaw_law={nclaw_law}")

    ident = stage_identify(material, dump=DUMPS / f"{material}_dataset_truth.npz",
                           tag=f"crossfloor_{material}{tag}", nclaw_law=nclaw_law)
    theta_rec = ident["theta_engine"]
    print(f"[floor] recovered {theta_rec} vs truth {theta_true}")

    rows = {}
    for scene in scenes:
        truth = DUMPS / f"{material}_{scene}_truth.npz"
        cloud = cloud_from_dump(truth)
        cells = {}
        for leg, theta in (("truth_theta", theta_true), ("recovered", theta_rec)):
            pred = OUT / f"{material}_{scene}_{leg}{tag}.npz"
            if not pred.exists():
                t0 = time.time()
                run_scene(material, scene, pred, theta=dict(theta), cloud=cloud,
                          nclaw_bc=nclaw_bc, nclaw_law=nclaw_law,
                          substeps=substeps)
                print(f"[floor] {scene}/{leg} simulated in {time.time() - t0:.0f}s")
            s = nclaw_position_mse(truth, pred)
            cells[leg] = {k: s[k] for k in
                          ("mse", "mse_final_frame", "rmse_mm", "n_frames")}
            print(f"[floor] {scene}/{leg}: MSE {s['mse']:.3e} "
                  f"(RMS {s['rmse_mm']:.2f} mm, {s['n_frames']} frames)")
        floor = cells["truth_theta"]["mse"]
        cells["identification_excess"] = (cells["recovered"]["mse"] / floor
                                          if floor > 0 else None)
        rows[scene] = cells

    res = {"material": material, "nclaw_bc": nclaw_bc, "nclaw_law": nclaw_law,
           "substeps": substeps,
           "theta_recovered": theta_rec,
           "theta_truth": theta_true,
           "identify_diagnostics": {k: ident.get(k) for k in
                                    ("refused_parameters",)},
           "scenes": rows}
    path = OUT / f"floor_{material}{tag}.json"
    path.write_text(json.dumps(res, indent=2, default=float))
    print(f"[floor] wrote {path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    sub = next((int(f.split("=", 1)[1]) for f in flags
                if f.startswith("--substeps=")), None)
    main(args[0] if args else "plasticine",
         nclaw_bc="--nclaw-bc" in flags, bisect="--bisect" in flags,
         nclaw_law="--nclaw-law" in flags, substeps=sub)
