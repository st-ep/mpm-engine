"""Lock the tabulated mu(I) material (fork id 13) to the Drucker-Prager kernel.

A recovered mu(I) CURVE is re-simulated through material 13 and scored against a
truth trajectory that material 2 (Drucker-Prager sand) generated, which only
means something if the two return maps agree when the curve is the constant the
Drucker-Prager cone represents. This measures that agreement instead of assuming
it, on a small column collapse: gravity only, a slip floor, and a runout that
friction alone sets.

Why they should agree, read off the kernels. Material 2 projects the Hencky
deviator to ||dev eps|| = -(3 lam + 2 mu) / (2 mu) tr(eps) alpha, which in stress
is tau_bar = 3 alpha p / sqrt(2) = mu_c p with mu_c = 3 alpha / sqrt(2), the
constant suite.friction_to_mu returns. Material 13 bisects
tau_bar_K - G dt gd - mu(I) p_K to zero and therefore lands on
tau_bar = mu(I) p_K, the same surface when mu(I) is that constant. The stress
branches coincide too: kirchoff_stress_drucker_prager forms
U diag(2 mu log sig + lam tr) V^T F^T, which for F = U diag(sig) V^T is exactly
kirchoff_stress_hencky.

The negative control carries as much weight as the match: halving the tabulated
friction must change the runout, or the test would pass on a material that never
reads its table.
"""
from __future__ import annotations

import numpy as np
import pytest

FRICTION_DEG = 25.0
STEPS = 1500
DT = 2.0e-4
N_GRID = 24


def _collapse(material: str, extra: dict) -> np.ndarray:
    """Final particle positions of a small column collapse on a slip floor."""
    import warp as wp

    wp.init()
    import torch

    from warpmpm.kernels import MPM_Simulator_WARP

    grid_lim = 1.0
    dx = grid_lim / N_GRID
    h = dx / 2.0
    axy = np.arange(-0.05, 0.05 + 0.5 * h, h)
    az = np.arange(0.0, 0.25 + 0.5 * h, h)
    pts = np.stack(np.meshgrid(axy, axy, az, indexing="ij"), -1).reshape(-1, 3)
    pts[:, 0] += 0.5
    pts[:, 1] += 0.5
    pts[:, 2] += 3.0 * dx + 0.01
    pts = np.ascontiguousarray(pts.astype(np.float32))
    vol0 = np.full(len(pts), h ** 3, dtype=np.float32)

    s = MPM_Simulator_WARP(len(pts), device="cpu")
    s.load_initial_data_from_torch(
        torch.from_numpy(pts), torch.from_numpy(vol0),
        n_grid=N_GRID, grid_lim=grid_lim, device="cpu")
    kw = {"material": material, "density": 1000.0, "E": 1.0e6, "nu": 0.2,
          "g": [0.0, 0.0, -9.8]}
    kw.update(extra)
    s.set_parameters_dict(kw, device="cpu")
    s.finalize_mu_lam(device="cpu")
    s.add_surface_collider((0.0, 0.0, 3.0 * dx), (0.0, 0.0, 1.0), "slip")
    for step in range(STEPS):
        s.p2g2p(step, DT, device="cpu")
    return s.export_particle_x_to_torch().cpu().numpy()


def _flat_table(mu: float) -> dict:
    """A constant mu(I) table on the dump-schema grid, log10 I in [-4, 0]."""
    return {"eta_table": np.full(256, float(mu)).tolist(),
            "eta_table_smin": -4.0, "eta_table_smax": 0.0,
            "grain_diameter": 1.0e-3, "grain_density": 1000.0}


def _runout(x: np.ndarray) -> float:
    return float(np.abs(x[:, :2] - 0.5).max())


@pytest.fixture(scope="module")
def drucker_prager() -> np.ndarray:
    return _collapse("sand", {"friction_angle": FRICTION_DEG})


def test_constant_table_matches_drucker_prager(drucker_prager):
    from experiments.nclaw.suite import friction_to_mu

    mu_c = friction_to_mu(FRICTION_DEG)
    x_tab = _collapse("tabulated_mu_i", _flat_table(mu_c))
    assert np.isfinite(x_tab).all() and np.isfinite(drucker_prager).all()
    spread = _runout(drucker_prager)
    assert spread > 0.1, f"the column barely collapsed (runout {spread:.3f} m)"
    gap = float(np.abs(drucker_prager - x_tab).max())
    assert gap < 1.0e-4, (
        f"tabulated mu(I) at the cone constant mu = {mu_c:.4f} drifts from "
        f"Drucker-Prager by {gap:.3e} m over a {spread:.3f} m runout")


def test_a_softer_table_changes_the_runout(drucker_prager):
    from experiments.nclaw.suite import friction_to_mu

    x_soft = _collapse("tabulated_mu_i", _flat_table(0.5 * friction_to_mu(FRICTION_DEG)))
    assert _runout(x_soft) > _runout(drucker_prager) + 0.01, (
        "halving the tabulated friction did not spread the column further, so "
        "the table is not being read"
    )
