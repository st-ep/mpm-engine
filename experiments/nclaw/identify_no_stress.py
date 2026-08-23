"""Identification without the stress channel, on NCLaw's own trajectories.

The cross-engine comparison in docs/four_method_comparison.md identified from
their trajectories with every stored channel available. This module runs the
same identification with the stress channel excluded: the stored kinematics
(x, v, L, F, volume, mass) are used exactly as before, and anything the
full-channel path read off the stress tensor is either replaced by a stated
model or refused.

What each material loses, measured on their dumps rather than assumed:

jelly, water
    nothing. The elastic momentum fit reads x, v, F, volume and mass, and the
    volumetric fit reads J from F, so neither ever touched the stress channel.
    Both are rerun at this tier to confirm that.
plasticine
    nothing in the elastic pair, and nothing in the plateau reading of the
    yield stress either, which takes the saturation of the deviatoric Hencky
    strain of the stored elastic F. This module adds the estimator that does
    not use that plateau: the momentum fit with one yield column on the
    fast-shearing set, reported next to it.
sand
    the pressure. The friction fit needs pressure as data, and the
    full-channel path read it from the 3D stress trace. Three sources are
    offered here and all three are reported:

    hencky_F      the Hencky volumetric relation on the stored elastic F at the
                  configured E and nu. This is how NCLaw's own
                  DruckerPragerPlasticity gets pressure (preset.py: E and nu
                  are held fixed and the friction angle is the fitted
                  parameter), so it is both stress-free and like-for-like with
                  their model. Their stress channel is elasticity(F) of the
                  same F, so this source is expected to reproduce the
                  full-channel pressure; the agreement is measured and
                  reported, never assumed.
    basal_scaled  pressure measured within one grid cell of the floor, which is
                  what a force plate under the pile gives, with the
                  depth-below-surface shape scaled to match that basal level.
                  This is the one variant that reads the stress channel, and it
                  reads it only inside that band.
    depth         the pure closure, density times gravity times depth below the
                  per-column free surface, from positions alone.

The deviatoric stress the yield-set gate and the cone-plateau estimator need is
reconstructed from F for the hencky_F source, by the same relation and the same
fixed elastic pair. The two closure sources supply pressure only, so their fits
gate the yield set on kinematics and have no plateau to fall back on.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIG_FLOOR = 1.0e-6            # their preset.py clamps the singular values at 0.05
PRESSURE_SOURCES = ("hencky_F", "basal_scaled", "depth")


# ---------------------------------------------------------------------------
# Stress-free pressure from the stored deformation gradient
# ---------------------------------------------------------------------------

def hencky_stress_parts(F: np.ndarray, E: float, nu: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cauchy pressure, deviatoric Cauchy stress and J from F at fixed (E, nu).

    The engine's ``kirchoff_stress_drucker_prager`` returns
    U center V^T F^T with center_ii = (2 mu log sig_i + lam sum log sig) / sig_i,
    which multiplies out to the Kirchhoff stress

        tau = U diag(2 mu eps_i + lam tr eps) U^T,   eps = log sig(F),

    the same Hencky form NCLaw's SigmaElasticity uses. Hence

        tr tau = (2 mu + 3 lam) tr eps,   dev tau = 2 mu U diag(eps_hat) U^T,

    and Cauchy is tau / J. Only the elastic pair enters, and for their sand it
    is a configured constant, not a fitted quantity.
    """
    from ident.weakform.elastic_grid import E_nu_to_moduli
    mu, lam = E_nu_to_moduli(float(E), float(nu))
    U, sig, _ = np.linalg.svd(np.asarray(F, dtype=float))
    eps = np.log(np.clip(np.abs(sig), SIG_FLOOR, None))
    tr = eps.sum(axis=-1)
    J = np.prod(np.clip(np.abs(sig), SIG_FLOOR, None), axis=-1)
    p = -(2.0 * mu + 3.0 * lam) * tr / (3.0 * J)
    eps_hat = eps - (tr / 3.0)[..., None]
    dev = (2.0 * mu / J)[..., None, None] * np.einsum(
        "...ij,...j,...kj->...ik", U, eps_hat, U)
    return p, dev, J


def column_surface_pressure(x: np.ndarray, rho: float, g_z: float, cell: float
                            ) -> np.ndarray:
    """rho |g| (h - z) with h the free surface of the particle column at (x, y).

    The closure this tree has used for granular collapses, here on the evolving
    cloud: bin the particles by cell in the two horizontal directions, take the
    highest particle of each bin as that column's surface, and read the depth
    below it. Positions alone; no stress and no deformation gradient.
    """
    x = np.asarray(x, dtype=float)
    T, P = x.shape[0], x.shape[1]
    p = np.zeros((T, P))
    gz = abs(float(g_z))
    for f in range(T):
        ix = np.floor(x[f, :, 0] / cell).astype(np.int64)
        iy = np.floor(x[f, :, 1] / cell).astype(np.int64)
        ix -= ix.min()
        iy -= iy.min()
        bid = ix * (iy.max() + 1) + iy
        surf = np.full(int(bid.max()) + 1, -np.inf)
        np.maximum.at(surf, bid, x[f, :, 2])
        p[f] = rho * gz * np.maximum(surf[bid] - x[f, :, 2], 0.0)
    return p


def basal_scaled_pressure(x: np.ndarray, p_measured: np.ndarray, floor_z: float,
                          cell: float, rho: float, g_z: float
                          ) -> tuple[np.ndarray, dict[str, Any]]:
    """The depth shape scaled per frame to the pressure measured at the base.

    A force plate under the pile measures pressure where the material touches
    it and nowhere else. ``p_measured`` is therefore read only inside one cell
    of the floor; the shape of the field above the band is the depth closure,
    and one scalar per frame lifts that shape onto the measured basal level. The
    scale is a ratio of medians over the band, which is why a band that holds
    too few particles falls back to the median scale of the frames that had
    enough rather than inventing one.
    """
    x = np.asarray(x, dtype=float)
    p_depth = column_surface_pressure(x, rho, g_z, cell)
    band = x[:, :, 2] <= floor_z + cell
    ok = band & np.isfinite(p_measured) & (p_measured > 0.0) & (p_depth > 0.0)
    scale = np.full(x.shape[0], np.nan)
    for f in range(x.shape[0]):
        sel = ok[f]
        if int(sel.sum()) >= 20:
            scale[f] = (np.median(p_measured[f][sel])
                        / max(np.median(p_depth[f][sel]), 1e-30))
    good = np.isfinite(scale)
    fallback = float(np.median(scale[good])) if good.any() else 1.0
    scale_used = np.where(good, scale, fallback)
    diag = {
        "band_cells": 1.0, "floor_z": float(floor_z),
        "n_frames_with_band_measurement": int(good.sum()),
        "n_frames": int(x.shape[0]),
        "scale_median": float(np.median(scale_used)),
        "scale_p05": float(np.percentile(scale_used, 5)),
        "scale_p95": float(np.percentile(scale_used, 95)),
        "band_particles_median": float(np.median(ok.sum(axis=1))),
        "fallback_scale": fallback,
    }
    return scale_used[:, None] * p_depth, diag


def pressure_agreement(p_model: np.ndarray, p_true: np.ndarray,
                       weight: np.ndarray | None = None) -> dict[str, float]:
    """How a pressure model compares with the stored stress trace. Diagnosis only.

    Reported so the bias of a closure is a number in the record rather than an
    expectation. Never fed back into any fit at this tier.
    """
    m = np.asarray(p_model, dtype=float)
    t = np.asarray(p_true, dtype=float)
    sel = np.isfinite(m) & np.isfinite(t) & (t > 0.0)
    if weight is not None:
        sel &= np.asarray(weight, dtype=bool)
    if not sel.any():
        return {"n": 0}
    ratio = m[sel] / t[sel]
    return {
        "n": int(sel.sum()),
        "ratio_median": float(np.median(ratio)),
        "ratio_p05": float(np.percentile(ratio, 5)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        "rel_err_median": float(np.median(np.abs(ratio - 1.0))),
        "p_model_median": float(np.median(m[sel])),
        "p_true_median": float(np.median(t[sel])),
    }


# ---------------------------------------------------------------------------
# Yield stress from the momentum balance, without the plateau
# ---------------------------------------------------------------------------

def hencky_yield_parts(F: np.ndarray, mu: float, lam: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The yield column of a von Mises solid: direction, volumetric part, norms.

    Returns N, the unit-norm flow direction U diag(eps_hat) U^T / ||eps_hat|| as
    a Cauchy-stress direction; sigma_vol, the mean Cauchy stress
    (2 mu / 3 + lam) tr(eps) / J that the elastic pair fixes as data; the norm
    ||eps_hat||; and J. With y the yield stress, the Cauchy stress of a particle
    at yield is sigma = sigma_vol I + y N / J, which is what makes y the single
    unknown of a linear momentum fit.
    """
    U, sig, _ = np.linalg.svd(np.asarray(F, dtype=float))
    s = np.clip(np.abs(sig), SIG_FLOOR, None)
    eps = np.log(s)
    tr = eps.sum(axis=-1)
    J = np.prod(s, axis=-1)
    eps_hat = eps - (tr / 3.0)[..., None]
    hat_norm = np.linalg.norm(eps_hat, axis=-1)
    N = np.einsum("...ij,...j,...kj->...ik", U,
                  eps_hat / np.maximum(hat_norm, 1e-30)[..., None], U)
    sigma_vol = (2.0 * mu / 3.0 + lam) * tr / J
    return N, sigma_vol, hat_norm, J


def identify_yield_momentum(arr: dict, mu_hat: float, lam_hat: float,
                            window_frames: int = 26, frame_stride: int = 2,
                            margin_cells: float = 3.0, gd_min: float = 1.0,
                            eps_gamma: float = 0.02, flow_frac_min: float = 0.9,
                            residual_bar: float = 0.15, log=print) -> dict:
    """von Mises yield stress as the one unknown of a momentum fit.

    At yield the return map holds the deviatoric Kirchhoff stress at norm
    ``yield_stress`` and leaves its direction alone, so with the elastic pair
    already recovered the stress of a flowing particle is

        tau = (2 mu / 3 + lam) tr(eps) I + y N,
        N = U diag(eps_hat) U^T / ||eps_hat||,   eps = log sig(F),

    one unknown y multiplying a known direction, with the volumetric part as
    data. The set that enters is the kinematically flowing one, particles whose
    equivalent shear rate clears ``gd_min``: a particle below yield carries
    elastic deviatoric stress this model does not describe, so it gates its
    nodes out instead of contributing.
    """
    from common.conventions import equivalent_shear_rate, sym
    from experiments.nclaw.suite import _longest_run, wall_planes
    from ident.weakform.elastic_grid import assemble_columns_timeweak, solve_elastic_grid

    N, p_vol, hat_norm, J = hencky_yield_parts(arr["F"], mu_hat, lam_hat)
    gd = equivalent_shear_rate(sym(arr["L"]), eps_gamma)
    eye = np.eye(3)[None]

    def columns_fn(f: int):
        finite = np.isfinite(N[f]).all(axis=(1, 2)) & np.isfinite(p_vol[f])
        flowing = finite & (gd[f] > gd_min) & (hat_norm[f] > 1e-9)
        if int(flowing.sum()) < 50:
            return None
        Vp = arr["volume"][f]
        w = np.where(flowing, Vp / np.maximum(J[f], 1e-30), 0.0)
        Vsig = w[:, None, None, None] * N[f][:, None, :, :]
        Vsig_known = np.where(flowing, Vp * p_vol[f], 0.0)[:, None, None] * eye
        return Vsig, Vsig_known, flowing, hat_norm[f]

    all_frames = list(range(0, arr["x"].shape[0], frame_stride))
    counts = [int((gd[f] > gd_min).sum()) for f in all_frames]
    frames = _longest_run(all_frames, counts, 50)
    if len(frames) < window_frames:
        return {"refused": True, "n_rows": 0, "n_rows_before_gating": 0,
                "reason": (f"only {len(frames)} contiguous frames with 50 flowing "
                           f"particles, fewer than the {window_frames}-frame window")}
    sysm = assemble_columns_timeweak(
        arr["x"], arr["v"], arr["mass"], arr["g"],
        arr["frame_dt"] * frame_stride, arr["n_grid"], arr["grid_lim"],
        columns_fn, n_columns=1, frames=frames, window_frames=window_frames,
        collider_planes=wall_planes(arr["n_grid"], arr["grid_lim"]),
        collider_margin_cells=margin_cells, valid_frac_min=flow_frac_min)
    if sysm.n_rows < 8:
        return {"refused": True, "reason": "no surviving rows",
                "n_rows": sysm.n_rows,
                "n_rows_before_gating": sysm.n_rows_before_gating}
    out = solve_elastic_grid(sysm)
    y = float(out["theta"][0])
    out.update({"yield_stress": y, "estimator": "momentum_yield_column",
                "refused": False, "gd_min": gd_min,
                "flow_frac_min": flow_frac_min,
                "n_flowing_frames": len(frames),
                "row_survival": sysm.row_survival,
                "n_rows_before_gating": sysm.n_rows_before_gating,
                "residual_bar": residual_bar,
                "hencky_dev_coverage": list(sysm.strain_coverage)})
    if not np.isfinite(y) or y <= 0.0 or float(out["residual_rel"]) > residual_bar:
        out.update({"refused": True,
                    "reason": (f"yield column fit y={y:.4e} at relative residual "
                               f"{float(out['residual_rel']):.3f} against a bar of "
                               f"{residual_bar}")})
    log(f"[ident] yield-momentum y={y:.4e} rows={sysm.n_rows} "
        f"resid={float(out['residual_rel']):.3f} refused={out['refused']}")
    return out


# ---------------------------------------------------------------------------
# The tier's identify stage
# ---------------------------------------------------------------------------

def _sand_pressures(arr: dict, material_truth: dict, basal_dump: Path | None,
                    log=print) -> dict[str, dict]:
    """The three pressure fields for the friction fit, each with its provenance."""
    from common.conventions import pressure_from_cauchy_3d_trace
    cell = arr["grid_lim"] / arr["n_grid"]
    E, nu = float(material_truth["E"]), float(material_truth["nu"])
    p_F, dev_F, _ = hencky_stress_parts(arr["F"], E, nu)
    out: dict[str, dict] = {
        "hencky_F": {
            "p": p_F, "dev": dev_F,
            "provenance": ("stored elastic F through the Hencky volumetric and "
                           f"deviatoric relation at the configured E={E:g}, nu={nu:g}; "
                           "no stress channel read"),
            "reads_stress": False,
        },
    }
    p_depth = column_surface_pressure(arr["x"], arr["meta"].rho_bulk, arr["g"][2], cell)
    out["depth"] = {
        "p": p_depth, "dev": None,
        "provenance": ("rho g times depth below the per-column free surface of the "
                       "position cloud, one cell columns; positions alone"),
        "reads_stress": False,
    }
    if basal_dump is not None:
        d = np.load(basal_dump)
        T, P = arr["x"].shape[0], arr["x"].shape[1]
        stress = d["stress"].astype(np.float64).reshape(T, P, 3, 3)
        p_true = pressure_from_cauchy_3d_trace(stress)
        floor_z = float(arr["meta"].extra.get("bound_cells", 3)) * cell
        p_basal, diag = basal_scaled_pressure(
            arr["x"], p_true, floor_z, cell, arr["meta"].rho_bulk, arr["g"][2])
        out["basal_scaled"] = {
            "p": p_basal, "dev": None, "diagnostics": diag,
            "provenance": ("stress trace read ONLY within one grid cell of the floor "
                           f"(z <= {floor_z:.4g} m), the depth-below-surface shape "
                           "scaled per frame to that basal level"),
            "reads_stress": True,
        }
        # diagnosis only: how each pressure model compares with the stored trace
        for key, entry in out.items():
            entry["agreement_with_stress_trace"] = pressure_agreement(
                entry["p"], p_true)
            log(f"[ident] pressure {key}: median ratio to the stress trace "
                f"{entry['agreement_with_stress_trace'].get('ratio_median')}")
    return out


def stage_identify_no_stress(material: str, dump: str | Path,
                             tag: str | None = None, nclaw_law: bool = False,
                             window_frames: int | None = None,
                             basal_dump: str | Path | None = None,
                             log=print) -> dict:
    """Identify from one trajectory with the stress channel excluded.

    Returns the same shape ``suite.stage_identify`` returns, so a runner can
    swap one for the other, plus ``theta_variants``: the extra estimators this
    tier makes available (sand's pressure sources, plasticine's momentum yield
    column), each already in the engine's own arguments so a rollout can be run
    per variant.
    """
    from experiments.nclaw import suite

    dump = Path(dump)
    if window_frames is None:
        window_frames = suite.WINDOW_FRAMES[material]
    arr = suite._load_arrays(dump)
    if arr["meta"].has_pressure:
        raise ValueError(
            f"{dump.name} still carries an oracle pressure "
            f"(pressure_source={arr['meta'].pressure_source!r}); run this on the "
            "tier dump written by experiments/nclaw/strip_channels.py so a leg "
            "that reads the stress trace cannot do so by accident")
    truth = suite.MATERIALS[material]["truth"]
    ident: dict[str, Any] = {
        "tier": arr["meta"].extra.get("tier", "no_stress"),
        "source_dump": dump.name,
        "n_grid": arr["n_grid"],
        "channel_provenance": arr["meta"].extra.get("channel_provenance", {}),
        "pressure_source_of_dump": arr["meta"].pressure_source,
    }
    variants: dict[str, dict] = {}
    walls: dict[str, float] = {}

    t0 = time.time()
    if material in ("jelly", "plasticine"):
        ident["elastic"] = suite.identify_elastic(
            arr, window_frames=window_frames, log=log)
        if material == "plasticine" and not ident["elastic"].get("refused", True):
            mu_hat, lam_hat = ident["elastic"]["mu"], ident["elastic"]["lam"]
            ident["yield"] = suite.identify_yield(arr, mu_hat, log=log)
            t1 = time.time()
            ident["yield_momentum"] = identify_yield_momentum(
                arr, mu_hat, lam_hat, window_frames=window_frames, log=log)
            walls["yield_momentum_s"] = time.time() - t1
            ym = ident["yield_momentum"]
            base, _ = suite.theta_for_engine(material, ident, nclaw_law=nclaw_law)
            variants["yield_momentum"] = {
                "theta": {**base,
                          "yield_stress": (truth["yield_stress"] if ym.get("refused", True)
                                           else ym["yield_stress"])},
                "refused_parameters": (["yield_stress"] if ym.get("refused", True)
                                       else []),
                "provenance": ("elastic pair from the momentum fit, yield stress from "
                               "the momentum fit with one yield column on the "
                               "kinematically flowing set"),
                "diagnostics": {k: ym.get(k) for k in
                                ("yield_stress", "n_rows", "residual_rel", "cond_AtA",
                                 "row_survival", "refused", "reason")},
            }
    elif material == "sand":
        fields = _sand_pressures(arr, truth, Path(basal_dump) if basal_dump else None,
                                 log=log)
        for key in PRESSURE_SOURCES:
            if key not in fields:
                continue
            entry = fields[key]
            t1 = time.time()
            fr = suite.identify_friction(
                arr, window_frames=window_frames, pressure=entry["p"],
                dev_stress=entry["dev"], pressure_label=key, log=log)
            walls[f"friction_{key}_s"] = time.time() - t1
            fr["pressure_provenance"] = entry["provenance"]
            fr["pressure_reads_stress"] = entry["reads_stress"]
            for extra in ("diagnostics", "agreement_with_stress_trace"):
                if extra in entry:
                    fr[f"pressure_{extra}"] = entry[extra]
            if key == "hencky_F":
                ident["friction"] = fr
            else:
                ident[f"friction_{key}"] = fr
                variants[key] = {
                    "theta": suite.theta_for_engine(
                        material, {"friction": fr}, nclaw_law=nclaw_law)[0],
                    "refused_parameters": suite.theta_for_engine(
                        material, {"friction": fr}, nclaw_law=nclaw_law)[1],
                    "provenance": entry["provenance"],
                    "diagnostics": {k: fr.get(k) for k in
                                    ("friction_angle", "friction_angle_solve",
                                     "mu_c", "mu_c_solve", "n_rows", "residual_rel",
                                     "cond_AtA", "refused", "reason")},
                }
    elif material == "water":
        ident["eos"] = suite.identify_eos(
            arr, window_frames=window_frames,
            form="linear" if nclaw_law else "power_law", log=log)
    walls["identify_total_s"] = time.time() - t0

    theta, refused = suite.theta_for_engine(material, ident, nclaw_law=nclaw_law)
    ident.update({
        "theta_engine": theta,
        "refused_parameters": refused,
        "theta_variants": variants,
        "wall_times_s": walls,
        "truth": truth,
        "nclaw_law": suite.MATERIALS[material].get("nclaw_law") if nclaw_law else None,
    })
    suite.OUT.mkdir(parents=True, exist_ok=True)
    name = (f"identify_no_stress_{material}.json" if tag is None
            else f"identify_no_stress_{material}_{tag}.json")
    (suite.OUT / name).write_text(json.dumps(ident, indent=2, default=float))
    log(f"[ident] no-stress theta={theta} refused={refused} "
        f"variants={sorted(variants)} in {walls['identify_total_s']:.1f}s")
    return ident
