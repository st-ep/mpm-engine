"""Validate the Bingham ELASTO-VISCOPLASTIC solid (fork "visplas", id 3) as a material.

Two claims to pin down before WorldBench's dough row can use the realization:

1. STATICS -- below yield the material is elastic, so a free-standing block must NOT
   creep (the regularized `newtonian.with_yield` FLUID realization creeps at a
   tau-independent, dt-limited rate -- the measured WorldBench Shaping--Bingham
   blocker). Measured side by side here.

2. FLOW LAW -- during steady plastic flow the material must behave as a Bingham fluid,
   and the fork-parameter conventions must be pinned: the Perzyna return
   (viscoplasticity_return_mapping_with_StVK) predicts steady simple shear
       tau = yield_stress / sqrt(3) + (plastic_viscosity / 2) * gamma_dot,
   i.e. a Bingham law with SHEAR yield stress tau_y and viscosity eta enters as
   yield_stress = sqrt(3)*tau_y, plastic_viscosity = 2*eta. Measured here with the
   validated 3D shear cell (shear_cell_3d.shear_segment): the top wall drags the block
   at a sweep of speeds, and per-particle dissipation rows give the pointwise
   constitutive line diss/gd = tau_y_eff + eta_eff * gd, fit by weighted least squares.

Run:  python experiments/bingham_evp_couette.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shear_cell_3d import GEOM, _build_block, shear_segment

from warpmpm import GridConfig, Solver, newtonian, vonmises

OUT = Path(__file__).resolve().parents[1] / "out" / "bingham_evp"
RHO = 1000.0


def evp(tau_y: float, eta: float, E: float = 2.0e5, nu: float = 0.3):
    """The Bingham EVP solid for a target SHEAR yield stress + Bingham viscosity,
    using the conversion this experiment validates."""
    return vonmises(E=E, nu=nu, yield_stress=np.sqrt(3.0) * tau_y, density=RHO
                    ).with_viscosity(2.0 * eta)


def statics(seconds: float = 1.0, dt: float = 2.0e-4, substeps: int = 4,
            device: str = "auto") -> dict:
    """Free-standing block under gravity: EVP dough dial vs the fluid realization of
    the same (tau_y, eta). Drift = top-height change + p95 particle displacement."""
    rows = {}
    for tag, mat in (
        ("evp tau_y=2000", evp(2000.0, 40.0)),
        ("evp tau_y=200", evp(200.0, 40.0)),
        ("fluid tau_y=2000", newtonian(eta=40.0, density=RHO, bulk_modulus=9.0e5
                                       ).with_yield(2000.0)),
    ):
        grid = GridConfig(n_grid=48, grid_lim=0.4)
        pos, vol, floor, _cx, _cy = _build_block(grid)
        # the fluid realization needs the finer stepping it uses in WorldBench shaping
        # (the yield term NaNs at the solid's dt); the EVP runs at the vonmises stepping
        d, sub = (5.0e-5, 16) if tag.startswith("fluid") else (dt, substeps)
        s = Solver(grid=grid, device=device).load_particles(pos.copy(), vol.copy())
        s.set_material(mat)
        s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
        n = round(seconds / (d * sub))
        for _ in range(n):
            s.step(d, sub)
        x = s.x()
        disp = np.linalg.norm(x - pos, axis=1)
        rows[tag] = dict(
            top_drop_mm=float((pos[:, 2].max() - x[:, 2].max()) * 1e3),
            disp_p95_mm=float(np.quantile(disp, 0.95) * 1e3),
            disp_max_mm=float(disp.max() * 1e3),
        )
        print(f"  statics {tag:18s}: top drop {rows[tag]['top_drop_mm']:6.2f} mm | "
              f"disp p95 {rows[tag]['disp_p95_mm']:6.2f} max "
              f"{rows[tag]['disp_max_mm']:6.2f} mm over {seconds:.1f} s", flush=True)
    return rows


def flow_fit(tau_y: float, eta: float, speeds, n_frames: int = 90,
             device: str = "auto") -> dict:
    """Shear the EVP block at a sweep of wall speeds; pool late-time per-particle
    dissipation rows and fit the pointwise Bingham line diss/gd = a + b*gd.
    Expect a ~ tau_y, b ~ eta under the sqrt(3)/2x conversion in evp()."""
    gds, taus, ws = [], [], []
    for v in speeds:
        seg = shear_segment(v, evp(tau_y, eta), n_frames=n_frames, dt=1.0e-4,
                            substeps=20, record_stress=True, device=device)
        rows = seg["strong_rows"][int(0.6 * len(seg["strong_rows"])):]  # steady window
        for gd, vol, diss in rows:
            m = gd > 0.25          # yielding particles only; diss/gd is noise at rest
            gds.append(gd[m])
            taus.append(diss[m] / gd[m])
            ws.append(vol[m] * gd[m] ** 2)   # dissipation weighting (trust flowing pts)
        print(f"  flow v={v:5.3f}: gd p50 {np.percentile(gds[-1], 50):5.2f}/s "
              f"({m.sum()} pts/frame)", flush=True)
    gd = np.concatenate(gds)
    tau = np.concatenate(taus)
    w = np.concatenate(ws)
    A = np.stack([np.ones_like(gd), gd], 1) * np.sqrt(w)[:, None]
    th, *_ = np.linalg.lstsq(A, tau * np.sqrt(w), rcond=None)
    a, b = float(th[0]), float(th[1])
    print(f"  fit over {len(gd)} pts: tau(gd) = {a:.0f} + {b:.1f}*gd  "
          f"(target {tau_y:.0f} + {eta:.1f}*gd | err {100*(a/tau_y-1):+.1f}% / "
          f"{100*(b/eta-1):+.1f}%)", flush=True)
    return dict(tau_y_fit=a, eta_fit=b, tau_y_target=tau_y, eta_target=eta,
                n_points=int(len(gd)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== Bingham EVP validation (block {GEOM}, shear cell) ===", flush=True)
    st = statics(device=args.device)
    # viscosity-visible dial (low yield) pins the slope; dough dial pins the intercept
    f_soft = flow_fit(200.0, 40.0, speeds=(0.05, 0.15, 0.35), device=args.device)
    f_dough = flow_fit(2000.0, 40.0, speeds=(0.15, 0.35), device=args.device)
    with (OUT / "bingham_evp.json").open("w") as f:
        json.dump(dict(statics=st, flow_soft=f_soft, flow_dough=f_dough), f, indent=2)
    print(f"wrote {OUT / 'bingham_evp.json'}  [{time.time()-t0:.0f}s]", flush=True)
