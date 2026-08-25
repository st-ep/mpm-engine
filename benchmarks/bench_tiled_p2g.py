"""Plain versus tiled P2G scatter: per-substep milliseconds on a 128^3 grid.

Both configurations run the split pipeline with the block sort on, so the only
difference is the scatter kernel. The p2g column is the ScopedTimer phase time
(synchronized, live launches); the wall column includes host work (sort keys,
block tables, the tick guard). The shared-memory win only shows on CUDA: the
CPU backend runs one lane per tile block, so the tiled numbers on CPU measure
correctness overhead, not the target speedup. Intended host: the GH200.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from warpmpm import GridConfig, Solver
from warpmpm.materials import newtonian


def scene(n: int, n_grid: int, device: str, tiled: bool) -> Solver:
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    rng = np.random.default_rng(0)
    edge = grid.dx * (n / 8.0) ** (1.0 / 3.0)  # 8 particles per cell
    lo = 0.5 * (grid.grid_lim - edge)
    pts = (rng.random((n, 3), dtype=np.float32) * edge + lo).astype(np.float32)
    vol = np.full(n, (grid.dx / 2.0) ** 3, np.float32)
    s = Solver(grid=grid, device=device, fused=False, sort_interval=1,
               tiled_p2g=tiled, profile=True).load_particles(pts, vol)
    s.set_material(newtonian(eta=2.0, density=1000.0, bulk_modulus=2.0e5))
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "separate", friction=0.3)
    s.add_domain_walls()
    v0 = np.zeros((n, 3), np.float32)
    v0[:, 2] = -0.5
    s.set_v(v0)
    return s


def bench(n: int, n_grid: int, device: str, tiled: bool, warmup: int,
          timed: int) -> dict:
    s = scene(n, n_grid, device, tiled)
    dt = 2.0e-5
    for _ in range(warmup):
        s.step(dt, 1)
    if device.startswith("cuda"):
        wp.synchronize_device(device)
    s._sim.time_profile = {}
    wall = np.empty(timed)
    for i in range(timed):
        t0 = time.perf_counter()
        s.step(dt, 1)
        if device.startswith("cuda"):
            wp.synchronize_device(device)
        wall[i] = time.perf_counter() - t0
    p2g_ms = float(np.median(s._sim.time_profile["p2g"]))
    nb = s._sim.tiled_p2g_blocks["n"] if tiled else 0
    return {"n": n, "path": "tiled" if tiled else "plain", "p2g_ms": p2g_ms,
            "wall_ms": float(np.median(wall)) * 1e3, "blocks": nb}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0",
                        help="Warp device, e.g. cuda:0 or cpu")
    parser.add_argument("--n", type=int, nargs="+", default=[100_000, 500_000])
    parser.add_argument("--n-grid", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed", type=int, default=50)
    args = parser.parse_args()

    print(f"device={args.device} grid={args.n_grid}^3 "
          f"median of {args.timed} substeps after {args.warmup} warmup")
    print(f"{'n':>9} {'path':>6} {'p2g ms':>9} {'wall ms':>9} {'blocks':>7}")
    for n in args.n:
        rows = [bench(n, args.n_grid, args.device, tiled, args.warmup, args.timed)
                for tiled in (False, True)]
        for r in rows:
            print(f"{r['n']:>9d} {r['path']:>6} {r['p2g_ms']:>9.3f} "
                  f"{r['wall_ms']:>9.2f} {r['blocks']:>7d}")
        print(f"{'':>9} {'ratio':>6} {rows[0]['p2g_ms'] / rows[1]['p2g_ms']:>9.2f}"
              f" {rows[0]['wall_ms'] / rows[1]['wall_ms']:>9.2f}")
