"""Function-encoder identification on NCLaw's own trajectories.

The same unknown-form dictionaries as the same-engine study (baseline.py),
pointed at the ingested cross-engine dumps and rolled out on their five scenes
with their metric, under the same engine-compatibility flags the known-form
cross-engine rows use (experiments.nclaw.compare). The opponent row at equal
task difficulty is NCLaw's learned network, which also fits a function rather
than a scalar from one trajectory; the known-form rows bound both from above.

Per material:
  sand        mu(I) through the trained granular basis, the accepted curve
              baked into the engine's tabulated material and rolled out.
  jelly       W'(I1bar) through the trained one-invariant basis plus one
              volumetric column. The engine has no tabulated hyperelastic
              material, so the rollout leg projects the recovered curve to its
              small-strain corotated pair and the leg name says so.
  water       the same hyperelastic family: for a fluid its deviatoric
              coefficients should return near zero and its volumetric column
              is exactly their linear law sigma = lam (J - 1) I, rolled out
              through the comparison EOS. The viscous family runs beside it
              and is expected to refuse.
  plasticine  every family this campaign ships is the wrong class for an
              elastoplastic solid, and the recorded refusals are the result.
              No trained plasticity basis is wired in here.

Run from the engine root:

    .venv/bin/python -m experiments.fe_ls.cross [material ...]

Writes out/fe_ls_cross/results.json and the prediction dumps beside it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.fe_ls.baseline import (
    VISCOUS_WINDOW,
    bake_mu_table,
    identify_friction_fe,
    identify_hyperelastic_fe,
    identify_viscous_fe,
    load_granular_fe,
)
from experiments.nclaw import suite

CROSS_DUMPS = suite.ROOT / "out" / "nclaw_cross_generalize" / "dumps"
OUT = suite.ROOT / "out" / "fe_ls_cross"
RESIDUAL_BAR = 0.15

# the engine-compatibility flags of the winning known-form rows, per material
FLAGS = {"jelly": {"nclaw_bc": True, "substeps": 1},
         "plasticine": {"nclaw_bc": True},
         "sand": {"nclaw_bc": True, "substeps": 1},
         "water": {"nclaw_bc": True, "nclaw_law": True}}

PUBLISHED = {"jelly": {"dataset": 2.4e-4, "time": 9.8e-4, "vel_0001": 2.4e-4,
                       "vel_0007": 2.4e-4, "vel_0008": 2.4e-4},
             "plasticine": {"dataset": 6.5e-5, "time": 1.4e-4, "vel_0001": 4.6e-5,
                            "vel_0007": 4.6e-5, "vel_0008": 4.6e-5},
             "sand": {"dataset": 2.6e-5, "time": 4.2e-5, "vel_0001": 6.5e-5,
                      "vel_0007": 6.5e-5, "vel_0008": 6.5e-5},
             "water": {"dataset": 2.0e-5, "time": 3.5e-4, "vel_0001": 1.9e-5,
                       "vel_0007": 1.9e-5, "vel_0008": 1.9e-5}}


def _load(material: str) -> dict:
    dump = CROSS_DUMPS / f"{material}_dataset_truth.npz"
    arr = suite._load_arrays(dump)
    # the ingested dumps carry empty law_params; the FE legs read truth values
    # from there only as reporting references, so fill them from the config
    if not arr["meta"].law_params:
        arr["meta"].law_params.update(suite.MATERIALS[material]["truth"])
    return arr


def _rollout_legs(material: str, legs: list[tuple[str, str, dict]],
                  log=print) -> dict:
    """Every accepted leg on every scene, seeded from their frame-0 clouds."""
    flags = FLAGS[material]
    scenes = sorted(p.name.split(f"{material}_", 1)[1].rsplit("_truth.npz", 1)[0]
                    for p in CROSS_DUMPS.glob(f"{material}_*_truth.npz"))
    rows: dict = {}
    for scene in scenes:
        truth = CROSS_DUMPS / f"{material}_{scene}_truth.npz"
        cloud = suite.cloud_from_dump(truth)
        n_expected = int(np.load(truth)["x"].shape[0])
        cells: dict = {}
        for leg, mat_key, theta in legs:
            pred = OUT / f"{material}_{scene}_{leg}.npz"
            if not pred.exists():
                t0 = time.time()
                suite.run_scene(mat_key, scene, pred, theta=dict(theta),
                                cloud=cloud,
                                nclaw_bc=flags.get("nclaw_bc", False),
                                nclaw_law=flags.get("nclaw_law", False),
                                substeps=flags.get("substeps"), log=log)
                log(f"[fe-cross] {scene}/{leg} simulated in {time.time()-t0:.0f}s")
            s = suite.nclaw_position_mse(truth, pred)
            cell = {k: s[k] for k in ("mse", "mse_final_frame", "rmse_mm", "n_frames")}
            cell["diverged"] = bool(s["n_frames"] < n_expected)
            if cell["diverged"]:
                cell["reason"] = (f"non-finite at frame {s['n_frames']} of "
                                  f"{n_expected}; partial score not comparable")
            cell["published"] = PUBLISHED[material][scene]
            if not cell["diverged"]:
                cell["margin_vs_published"] = PUBLISHED[material][scene] / s["mse"]
                label = f"{cell['margin_vs_published']:.1f}x vs published"
            else:
                label = "diverged"
            log(f"[fe-cross] {scene}/{leg}: MSE {s['mse']:.3e} ({label})")
            cells[leg] = cell
        rows[scene] = cells
    return rows


def run_material(material: str, log=print) -> dict:
    arr = _load(material)
    res: dict = {"material": material, "source": f"{material}_dataset_truth.npz",
                 "tier": "full_channels", "flags": FLAGS[material],
                 "identify": {}, "legs_rolled": [], "scenes": {}}
    legs: list[tuple[str, str, dict]] = []

    if material == "sand":
        fe, prior = load_granular_fe()
        t0 = time.time()
        ident = identify_friction_fe(arr, fe, prior, log=log)
        ident["wall_seconds"] = time.time() - t0
        res["identify"]["granular_fe"] = {
            k: ident.get(k) for k in
            ("refused", "reason", "residual_rel", "n_rows", "curve_error",
             "wall_seconds", "K")}
        if not ident.get("refused", True) and ident.get("baked_table"):
            b = ident["baked_table"]
            legs.append(("fe_mu_of_I", "sand_table",
                         {"eta_table": b["table"], "eta_table_smin": b["smin"],
                          "eta_table_smax": b["smax"]}))
            # control: a CONSTANT table at the true cone level through the same
            # tabulated material, separating curve error from the gap between
            # the tabulated return map and their Drucker-Prager integrator
            mu_true = suite.friction_to_mu(
                float(suite.MATERIALS["sand"]["truth"]["friction_angle"]))
            legs.append(("flat_table_truth_mu", "sand_table",
                         {"eta_table": [mu_true] * b["n_points"],
                          "eta_table_smin": b["smin"],
                          "eta_table_smax": b["smax"]}))
        res["identify"]["granular_fe_full"] = {
            k: v for k, v in ident.items() if k not in ("curve",)}

    elif material in ("jelly", "water"):
        from ident.weakform.elastic_grid import moduli_to_E_nu
        t0 = time.time()
        ident = identify_hyperelastic_fe(arr, log=log)
        ident["wall_seconds"] = time.time() - t0
        accepted = (not ident.get("refused", True)
                    and float(ident.get("residual_rel", 1.0)) <= RESIDUAL_BAR)
        res["identify"]["hyperelastic_fe"] = {
            k: ident.get(k) for k in
            ("refused", "residual_rel", "n_rows", "shear_modulus_from_curve",
             "shear_modulus_rel_err", "bulk_coefficient", "bulk_rel_err",
             "wall_seconds", "K")}
        res["identify"]["hyperelastic_fe"]["accepted"] = accepted
        if material == "jelly" and accepted:
            mu_h = float(ident["shear_modulus_from_curve"])
            lam_h = float(ident["bulk_coefficient"]) - 2.0 * mu_h / 3.0
            E_h, nu_h = moduli_to_E_nu(mu_h, lam_h)
            legs.append(("fe_projected_corotated", "jelly",
                         {"E": float(E_h), "nu": float(nu_h)}))
            res["identify"]["projection"] = {
                "note": ("the engine has no tabulated hyperelastic material; "
                         "the leg rolls the curve's small-strain corotated "
                         "pair, W1 -> mu/2 and volumetric -> lam + 2mu/3"),
                "E": float(E_h), "nu": float(nu_h)}
        if material == "water":
            # for a fluid the family's deviatoric part should vanish and the
            # volumetric column is exactly their law: lam is read directly
            lam_h = float(ident["bulk_coefficient"])
            res["identify"]["lam_from_volumetric_column"] = lam_h
            res["identify"]["lam_truth"] = 57692.0
            if accepted or float(ident.get("residual_rel", 1.0)) <= 0.5:
                legs.append(("fe_volumetric_eos", "water",
                             {"E": suite.lam_to_E(lam_h, 0.3), "nu": 0.3}))
            t0 = time.time()
            visc = identify_viscous_fe(
                arr, window_frames=VISCOUS_WINDOW["water"], log=log)
            visc["wall_seconds"] = time.time() - t0
            res["identify"]["viscous_fe"] = {
                k: visc.get(k) for k in
                ("refused", "reason", "residual_rel", "wall_seconds")}

    elif material == "plasticine":
        t0 = time.time()
        visc = identify_viscous_fe(
            arr, window_frames=VISCOUS_WINDOW["plasticine"], log=log)
        visc["wall_seconds"] = time.time() - t0
        res["identify"]["viscous_fe"] = {
            k: visc.get(k) for k in
            ("refused", "reason", "residual_rel", "wall_seconds")}
        t0 = time.time()
        hyper = identify_hyperelastic_fe(arr, log=log)
        hyper["wall_seconds"] = time.time() - t0
        res["identify"]["hyperelastic_fe"] = {
            k: hyper.get(k) for k in
            ("refused", "residual_rel", "n_rows", "shear_modulus_from_curve",
             "shear_modulus_rel_err", "bulk_coefficient", "wall_seconds")}
        res["identify"]["hyperelastic_fe"]["accepted"] = (
            not hyper.get("refused", True)
            and float(hyper.get("residual_rel", 1.0)) <= RESIDUAL_BAR)
        res["note"] = ("no trained plasticity basis is wired into this "
                       "campaign; the family refusals above are the result "
                       "for this material")

    res["legs_rolled"] = [leg for leg, _, _ in legs]
    if legs:
        res["scenes"] = _rollout_legs(material, legs, log=log)
    return res


def main(argv: list[str] | None = None) -> None:
    mats = (argv if argv is not None else sys.argv[1:]) or \
        ["sand", "jelly", "water", "plasticine"]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for m in mats:
        results[m] = run_material(m)
        path.write_text(json.dumps(results, indent=2, default=float))
        print(f"[fe-cross] {m} recorded -> {path}")


if __name__ == "__main__":
    main()
