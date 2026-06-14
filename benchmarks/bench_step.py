"""Baseline benchmark: ms per p2g2p step for the dough material, at several resolutions.

This is the number that anchors the whole performance plan (sparse + implicit are judged
against it). Warp CPU is single-threaded on Apple Silicon, so this measures the explicit
dense baseline we must beat.
"""
from __future__ import annotations

import time

from warpmpm import GridConfig, Solver
from warpmpm.scenes import block, dough


def bench(n_grid: int, n_warm: int = 5, n_timed: int = 30) -> dict:
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    pos, vol, floor = block(grid, size=(0.12, 0.05, 0.06), ppc=2)
    s = Solver(grid=grid).load_particles(pos, vol)
    s.set_material(dough())
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    dt = 2.0e-5
    s.step(dt, n_warm)  # warm (kernel JIT compile)
    t0 = time.time()
    s.step(dt, n_timed)
    ms = 1e3 * (time.time() - t0) / n_timed
    return {"n_grid": n_grid, "n_particles": s.n_particles, "ms_per_step": ms}


if __name__ == "__main__":
    print(f"{'n_grid':>7} {'n_particles':>12} {'ms/step':>10}")
    for ng in (48, 64, 72):
        r = bench(ng)
        print(f"{r['n_grid']:>7d} {r['n_particles']:>12d} {r['ms_per_step']:>10.2f}")
