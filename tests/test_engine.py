"""Foundational engine tests: packaging, no-NaN stepping, mass/volume conservation."""
from __future__ import annotations

import numpy as np

from warpmpm import GridConfig, Solver
from warpmpm.scenes import block, dough


def _dough_solver(n_grid: int = 40):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    pos, vol, floor = block(grid, size=(0.10, 0.05, 0.05), ppc=2)
    s = Solver(grid=grid).load_particles(pos, vol)
    s.set_material(dough())
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    return s


def test_packaging_import_and_step():
    # imports work with no caller sys.path manipulation, and a tiny sim steps cleanly
    s = _dough_solver(32)
    assert s.n_particles > 100
    s.step(2.0e-5, 5)
    assert np.isfinite(s.x()).all()


def test_no_nan_under_gravity():
    s = _dough_solver(40)
    s.step(2.0e-5, 50)
    assert np.isfinite(s.x()).all()
    assert np.isfinite(s.stress()).all()


def test_particle_count_conserved():
    s = _dough_solver(40)
    n0 = s.n_particles
    s.step(2.0e-5, 50)
    assert s.x().shape[0] == n0  # MPM particles are persistent material points


def test_volume_weakly_conserved():
    # dough is weakly compressible (EOS); total volume should drift only a few percent.
    # F is uninitialized until the first p2g2p, so warm up before measuring det(F).
    s = _dough_solver(40)
    s.step(2.0e-5, 5)
    F0 = np.abs(np.linalg.det(s.F())).mean()
    s.step(2.0e-5, 100)
    F1 = np.abs(np.linalg.det(s.F())).mean()
    rel = abs(F1 - F0) / F0
    assert rel < 0.05, f"volume drifted {rel:.1%} (expected < 5%)"


def test_settles_downward():
    # under gravity on a sticky floor the blob's centroid should not rise
    s = _dough_solver(40)
    z0 = s.x()[:, 2].mean()
    s.step(2.0e-5, 200)
    z1 = s.x()[:, 2].mean()
    assert z1 <= z0 + 1e-4
