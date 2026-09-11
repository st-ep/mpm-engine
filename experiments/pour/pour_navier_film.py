"""Verify the experimental Navier boundary against analytical Newtonian film flow."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from warpmpm import GridConfig, Solver, newtonian
from experiments.pour.pour_film_benchmark import plane_sdf
from experiments.pour.pour_navier_wall import install_navier_wall

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    length, height, density, bulk = args.extent, args.height, 1260., 9e5
    grid = GridConfig(n_grid=args.grid, grid_lim=length)
    dx, hp = grid.dx, grid.dx / 2
    bottom = length / 4 + args.phase * dx
    y0, y1 = length / 2 - 2 * dx, length / 2 + 2 * dx
    axes = [np.arange(a + hp / 2, b - hp / 8, hp)
            for a, b in [(0, length), (y0, y1), (bottom, bottom + height)]]
    pos = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    vol = np.full(len(pos), hp**3, dtype=np.float32)
    gx, gz = 9.81 * np.sin(np.radians(args.angle)), -9.81 * np.cos(np.radians(args.angle))
    solver = Solver(grid=grid, device=args.device, periodic_x=True).load_particles(pos.astype(np.float32), vol)
    solver.set_material(newtonian(eta=args.eta, density=density, bulk_modulus=bulk), g=[gx, 0., gz])
    wall = solver.add_sdf_collider(plane_sdf(length, bottom), center=(0, 0, 0),
                                  band=.5 * dx, surface="separable", friction=0.)
    installer = install_navier_wall
    if args.method == "robin":
        from experiments.pour.pour_navier_robin_wall import install_navier_robin_wall
        installer = install_navier_robin_wall
    if args.method == "quadratic":
        from experiments.pour.pour_navier_quadratic_wall import install_navier_quadratic_wall
        installer = install_navier_quadratic_wall
    if args.method == "wet-traction":
        from experiments.pour.pour_navier_traction_wall import install_navier_traction_wall
        installer = install_navier_traction_wall
    method = installer(solver, wall, slip_length_m=args.slip_mm * .001, eta_pa_s=args.eta)
    solver.add_plane((0, y0, 0), (0, 1, 0), "slip")
    solver.add_plane((0, y1, 0), (0, -1, 0), "slip")
    limit = min(dx / np.sqrt(bulk / density), density * dx**2 / (6 * args.eta))
    nominal_dt = .2 * limit
    if args.production_step:
        production_limit = min(.28 * dx / np.sqrt(1.1 * bulk / density),
                               density * dx**2 / (6 * args.eta))
        nominal_dt = (1/60) / np.ceil((1/60) / production_limit)
    steps = int(np.ceil(args.duration / (nominal_dt * args.dt_scale)))
    dt = args.duration / steps
    out = args.output / f"b{args.slip_mm:g}_n{args.grid}_phase{args.phase:g}_dt{args.dt_scale:g}"
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(**method, n_grid=args.grid, dx_m=dx, phase_cells=args.phase,
                    height_m=height, density=density, angle_deg=args.angle,
                    duration_s=args.duration, dt_s=dt, production_step=args.production_step,
                    extent_m=length, particle_count=len(pos), robot_outcomes_used=[],
                    input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), ROOT / "experiments/pour/pour_navier_wall.py",
                                            ROOT / "experiments/pour/pour_navier_robin_wall.py",
                                            ROOT / "experiments/pour/pour_navier_quadratic_wall.py",
                                            ROOT / "experiments/pour/pour_navier_traction_wall.py"]})
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    exact_mean = density * gx / args.eta * (height**2 / 3 + args.slip_mm * .001 * height)
    start, done, history = time.monotonic(), 0, []
    while done < steps:
        count = min(max(1, steps // 100), steps - done)
        solver.step(dt, count)
        done += count
        x, v = solver.x(), solver.v()
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError("Nonfinite film state")
        history.append([done * dt, v[:, 0].mean()])
    history = np.array(history)
    z = x[:, 2] - bottom
    exact = density * gx / args.eta * (height * z - z*z/2 + args.slip_mm * .001 * height)
    result = dict(**protocol, mean_velocity_m_s=float(v[:, 0].mean()),
                  analytical_mean_velocity_m_s=float(exact_mean),
                  flux_ratio=float(v[:, 0].mean() / exact_mean),
                  profile_relative_l2=float(np.linalg.norm(v[:, 0] - exact) / np.linalg.norm(exact)),
                  tail_change_relative=float(np.ptp(history[history[:, 0] >= .8 * args.duration, 1]) / exact_mean),
                  minimum_height_m=float(z.min()), maximum_height_m=float(z.max()),
                  elapsed_s=time.monotonic() - start)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out / "state.npz", x=x, v=v, history=history, exact_profile=exact, bottom=bottom)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(history[:, 0], history[:, 1]); ax[0].axhline(exact_mean, color="k", ls="--")
    ax[0].set(xlabel="Time (s)", ylabel="Mean velocity (m/s)")
    bins = np.arange(0, height + hp / 2, hp)
    means = [np.mean(v[(z >= a) & (z < b), 0]) if np.any((z >= a) & (z < b)) else np.nan
             for a, b in zip(bins[:-1], bins[1:])]
    zz = np.linspace(0, height, 100)
    ax[1].plot(density * gx / args.eta * (height*zz - zz*zz/2 + args.slip_mm*.001*height), zz*1000,
               "k--", label="Analytical Navier film")
    ax[1].plot(means, (bins[1:] + bins[:-1])*500, "o-", label="MPM")
    ax[1].set(xlabel="Velocity (m/s)", ylabel="Height (mm)"); ax[1].legend(fontsize=8)
    fig.suptitle(out.name); fig.tight_layout(); fig.savefig(out / "profile.png", dpi=150); plt.close(fig)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--extent", type=float, default=.032)
    p.add_argument("--height", type=float, default=.008)
    p.add_argument("--production-step", action="store_true")
    p.add_argument("--phase", type=float, default=0.)
    p.add_argument("--slip-mm", type=float, default=1.)
    p.add_argument("--eta", type=float, default=3.4392377844275503)
    p.add_argument("--angle", type=float, default=60.)
    p.add_argument("--duration", type=float, default=.18)
    p.add_argument("--dt-scale", type=float, default=1.)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--method", choices=["traction", "robin", "quadratic", "wet-traction"], default="traction")
    p.add_argument("--output", type=Path, default=ROOT / "out/pour_navier_calibration/film_v1")
    run(p.parse_args())
