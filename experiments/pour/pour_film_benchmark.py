"""Independent MPM verification against steady Newtonian film flow.

A streamwise-periodic layer starts at rest under tilted gravity. The exact
no-slip/shear-free solution is rho*g*sin(alpha)*(h*z-z*z/2)/eta. Neither the
layer height nor viscosity is adjusted to simulation or robot outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from warpmpm import GridConfig, Solver, newtonian
from warpmpm.geometry.mesh_sdf import SDFData

ROOT = Path(__file__).resolve().parents[2]


def steady_profile(z, height, eta, density, tangential_gravity):
    return density * tangential_gravity * (height * z - z**2 / 2) / eta


def plane_sdf(length, bottom):
    # The SDF API requires a closed solid with positive field margins. A wide
    # box has its top exactly at the benchmark plane, with every edge outside
    # the periodic fluid domain. The field is linear in the fluid contact region.
    origin = np.full(3, -length)
    res = 97
    cell = 3 * length / (res - 1)
    axes = [origin[i]+cell*np.arange(res) for i in range(3)]
    xyz = np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
    center = np.array([length/2,length/2,bottom-.008])
    half = np.array([length,length,.008])
    q = np.abs(xyz-center)-half
    values = np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(np.max(q,axis=-1),0)
    grads = np.stack(np.gradient(values,cell),axis=-1)
    return SDFData(values.astype(np.float32), grads.astype(np.float32), origin,
                   cell, float(np.max(np.abs(values))))


def run(args):
    length, height, density, bulk = .032, .008, 1260., 9e5
    grid = GridConfig(n_grid=args.grid, grid_lim=length)
    dx, hp = grid.dx, grid.dx / 2
    bottom = .008 + args.phase * dx
    y0, y1 = length / 2 - 2 * dx, length / 2 + 2 * dx
    axes = [np.arange(a + hp / 2, b - hp / 8, hp)
            for a, b in [(0, length), (y0, y1), (bottom, bottom + height)]]
    pos = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    vol = np.full(len(pos), hp**3, dtype=np.float32)
    assert abs(len(axes[2]) * hp - height) < 1e-10
    angle = np.radians(args.angle)
    gx, gz = 9.81 * np.sin(angle), -9.81 * np.cos(angle)
    solver = Solver(grid=grid, device=args.device, periodic_x=True)
    solver.load_particles(pos.astype(np.float32), vol)
    solver.set_material(newtonian(eta=args.eta, density=density, bulk_modulus=bulk),
                        g=[gx, 0., gz])
    friction = 0. if args.surface == 'sticky' else .05
    if args.surface == 'ghost':
        if args.collider != 'plane':
            raise ValueError('The experimental ghost wall supports only the flat plane')
        from experiments.pour.pour_ghost_wall_benchmark import add_ghost_plane
        add_ghost_plane(solver,bottom)
    elif args.collider == 'plane':
        solver.add_plane((0, 0, bottom), (0, 0, 1),
                         'sticky' if args.surface == 'sticky' else 'separate',
                         friction=friction)
    else:
        solver.add_sdf_collider(plane_sdf(length, bottom), center=(0, 0, 0),
                                band=args.band_cells * dx, surface=args.surface,
                                friction=friction)
    solver.add_plane((0, y0, 0), (0, 1, 0), 'slip')
    solver.add_plane((0, y1, 0), (0, -1, 0), 'slip')
    # Acoustic and explicit-viscous stability constraints, not a fitted time step.
    limit = min(dx / np.sqrt(bulk / density), density * dx**2 / (6 * args.eta))
    nsteps = int(np.ceil(args.duration / (.2 * limit)))
    dt = args.duration / nsteps
    batch = max(1, nsteps // 120)
    name = f'{args.collider}_{args.surface}_n{args.grid}_phase{args.phase:g}_band{args.band_cells:g}'
    out = args.output / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError(f'Retain earlier result; choose another output directory: {out}')
    print(f'{name}: N={len(pos)}, h/dx={height/dx:g}, steps={nsteps}, dt={dt:.3g}', flush=True)
    history, done, started = [], 0, time.monotonic()
    while done < nsteps:
        count = min(batch, nsteps - done)
        solver.step(dt, count); done += count
        x, v = solver.x(), solver.v()
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise RuntimeError('Nonfinite film state')
        history.append([done * dt, float(v[:, 0].mean()), float(v[:, 0].std()),
                        float(x[:, 2].min() - bottom), float(x[:, 2].max() - bottom)])
        if len(history) % 30 == 0:
            print(f't={done*dt:.4f}: mean u={history[-1][1]:.6f} m/s', flush=True)
    history = np.asarray(history)
    z = x[:, 2] - bottom
    exact = steady_profile(z, height, args.eta, density, gx)
    exact_mean = density * gx * height**2 / (3 * args.eta)
    mean_velocity = float(v[:, 0].mean())
    tail = history[history[:, 0] >= .8 * args.duration, 1]
    result = dict(name=name, collider=args.collider, surface=args.surface,
                  grid=args.grid, dx_m=dx, phase=args.phase, band_cells=args.band_cells,
                  eta_pa_s=args.eta, density=density, height_m=height,
                  angle_deg=args.angle, duration_s=args.duration, dt_s=dt,
                  particle_count=len(pos), mean_velocity_m_s=mean_velocity,
                  exact_noslip_mean_velocity_m_s=float(exact_mean),
                  flux_ratio_to_noslip=mean_velocity / exact_mean,
                  profile_relative_l2=float(np.linalg.norm(v[:, 0] - exact) / np.linalg.norm(exact)),
                  tail_change_relative=float(np.ptp(tail) / exact_mean),
                  z_min_m=float(z.min()), z_max_m=float(z.max()),
                  max_transverse_speed_m_s=float(np.max(np.linalg.norm(v[:, 1:], axis=1))),
                  elapsed_s=time.monotonic() - started, robot_outcomes_used=[],
                  interpretation='Verification benchmark; separable wall is not expected to match no-slip theory',
                  input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), Path(__file__).with_name('pour_ghost_wall_benchmark.py'),
                                          *sorted((ROOT/'src/warpmpm').rglob('*.py'))]})
    np.savez_compressed(out / 'state.npz', x=x, v=v, history=history, bottom=bottom,
                        exact_profile=exact, nominal_height=height)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    bins = np.linspace(0, height, 2 * round(height / dx) + 1)
    centers = .5 * (bins[1:] + bins[:-1])
    means = [float(v[(z >= a) & (z < b), 0].mean())
             if np.any((z >= a) & (z < b)) else np.nan for a, b in zip(bins[:-1], bins[1:])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history[:, 0], history[:, 1], label='MPM mean velocity')
    axes[0].axhline(exact_mean, color='k', ls='--', label='No-slip analytical mean')
    axes[0].set(xlabel='Time (s)', ylabel='Velocity (m/s)'); axes[0].legend(fontsize=8)
    zz = np.linspace(0, height, 100)
    axes[1].plot(steady_profile(zz, height, args.eta, density, gx), zz*1000, 'k--', label='No-slip theory')
    axes[1].plot(means, centers*1000, 'o-', ms=3, label='MPM')
    axes[1].set(xlabel='Downslope velocity (m/s)', ylabel='Height above physical wall (mm)')
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle(name); fig.tight_layout(); fig.savefig(out/'profile.png', dpi=150); plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, default=32)
    parser.add_argument('--surface', choices=['sticky', 'separable', 'ghost'], default='sticky')
    parser.add_argument('--collider', choices=['plane', 'sdf'], default='plane')
    parser.add_argument('--phase', type=float, default=0.)
    parser.add_argument('--band-cells', type=float, default=0.)
    parser.add_argument('--eta', type=float, default=3.397435009739053)
    parser.add_argument('--angle', type=float, default=60.)
    parser.add_argument('--duration', type=float, default=.12)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=ROOT/'out/pour_physics_audit/film')
    run(parser.parse_args())
