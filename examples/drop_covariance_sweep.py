"""Sweep DROP HEIGHT and plot parameter covariance -- the free-flight twin of
pressure_covariance_sweep.py (same material, same estimator, same four figures;
the action knob is the release height of a falling ball of dough instead of the
plate stop-pressure).

A ball of the Bingham dough (tau_y=200 Pa, eta=40 Pa s) is released just above
the sticky floor with the impact speed of a drop from height h, v0 = sqrt(2 g h)
(free flight in vacuum is exact, so dead air time is not simulated). On impact
the ball splats: gravity + kinetic energy convert to plastic dissipation, and
the scalar power balance

    P_grav - dKE/dt  =  tau_y * X1 + eta * X2

is least-squares fit per tick exactly as in the press sweep -- the same
_fit_power_balance is imported, with the tool term F*v identically zero.

Why height is the action that selects the constants: yielding needs the
inertial stress rho v^2 to reach tau_y, i.e. h >~ tau_y / (2 rho g) ~ 1 cm.
Below that the ball plops and creeps (X1, X2 ~ 0, impact power ~ the assumed
sensor noise sigma_power): the information matrix A^T A is tiny and the
covariance blows up -- the wrong-action regime. Above it the splat sweeps
strain rates from ~ v/R down to 0 and both constants sharpen with h.

What the drop determines -- and what it cannot: a splat probes strain rates
~ v/R down to 0, so it is an ETA experiment (sigma_eta collapses to <1% by
h ~ 0.1 m and eta_hat lands within ~10-20%). tau_y sits close to the splat's
null direction: its energy-route estimate rides on the unmodeled losses
(sticky-floor absorption of the spreading pancake, EOS compression work, grid
transfer) and is model-error dominated -- ~-45% at 128^3 and SIGN-UNSTABLE at
192^3 -- while sigma(tau_y) keeps shrinking. The slow press
(pressure_covariance_sweep.py) is the tau_y experiment; the pair demonstrates
action-dependent identifiability, and the covariance ratio flags it:
sigma_tau/sigma_eta ~ 16 at h = 0.2 m.

Valid regime and the deliberate breakdown: the scalar energy route holds up to
h ~ 0.2 m (rho v^2 ~ 20 tau_y). The default ladder continues to 0.5-1 m ON
PURPOSE: there the BC/transfer losses dominate the yield channel and tau_y_hat
inflates to 650-2300 Pa while its sigma shrinks to 2-4% -- the overconfidence
exhibit (a small covariance never certifies the parameters against model
error; h = 1 m additionally splashes, raining droplets onto the sticky floor
for the whole window).

Run:
  python examples/drop_covariance_sweep.py --device cuda:0
  python examples/drop_covariance_sweep.py --device cuda:0 --seed 0   # quick single-seed run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, newtonian

import pressure_covariance_sweep as pcs

OUT = Path(__file__).resolve().parents[1] / "out" / "drop_covariance_sweep"
G_MAG = pcs.G_MAG
BOTTOM_GAP = 1.0e-3   # m; release gap folded into the equivalent height below


def sample_ball(grid: GridConfig, radius: float, ppc: int, seed: int,
                bottom_gap: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Jittered-lattice ball resting bottom_gap above the floor (block() conventions:
    lattice pitch dx/ppc, per-particle volume h^3, floor at 3 dx)."""
    h = grid.dx / ppc
    floor = 3.0 * grid.dx
    c = np.array([grid.grid_lim * 0.5, grid.grid_lim * 0.5, floor + bottom_gap + radius])
    n = int(np.ceil(radius / h))
    ax = c[0] + h * (np.arange(-n, n + 1) + 0.5)
    ay = c[1] + h * (np.arange(-n, n + 1) + 0.5)
    az = c[2] + h * (np.arange(-n, n + 1) + 0.5)
    g = np.stack(np.meshgrid(ax, ay, az, indexing="ij"), axis=-1).reshape(-1, 3)
    g = g[np.linalg.norm(g - c, axis=1) <= radius]
    rng = np.random.default_rng(seed)
    g = g + rng.uniform(-0.25 * h, 0.25 * h, size=g.shape)
    pos = g.astype(np.float32)
    vol = np.full(len(pos), h**3, dtype=np.float32)
    return pos, vol, floor


def run_drop(
    seed: int,
    height: float,
    *,
    device: str,
    n_grid: int,
    ticks: int,
    substeps: int,
    dt: float,
    radius: float,
    ppc: int,
    tau_y: float,
    eta: float,
    density: float,
    bulk: float,
) -> tuple[list[dict], dict]:
    """One ball drop from equivalent height `height`; returns (per-tick records, meta)."""
    if height <= BOTTOM_GAP:
        raise ValueError(f"height {height} must exceed the release gap {BOTTOM_GAP}")
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    pos, vol0, floor = sample_ball(grid, radius, ppc, seed, BOTTOM_GAP)
    solver = Solver(grid=grid, device=device).load_particles(pos, vol0)
    solver.set_material(newtonian(eta=eta, density=density, bulk_modulus=bulk).with_yield(tau_y))
    solver.add_plane((0, 0, floor), (0, 0, 1), "sticky")

    v_impact = float(np.sqrt(2.0 * G_MAG * (height - BOTTOM_GAP)))
    vel = np.zeros_like(pos)
    vel[:, 2] = -v_impact
    solver.set_v(vel)

    mass = float(density * vol0.sum())
    dt_ctrl = dt * substeps
    records = []
    for tick in range(ticks + 1):
        if tick > 0:
            solver.step(dt, substeps)
        v = solver.v()
        L = solver.L()
        vol = solver.vol()
        gd, q = pcs.equivalent_shear_rate(L)
        ke = float(0.5 * density * np.sum(vol * np.sum(v * v, axis=1)))
        records.append({
            "tick": tick,
            "t": tick * dt_ctrl,
            # no tool in this experiment: the F*v power term is identically zero,
            # which _fit_power_balance handles as F_grid * down_speed = 0
            "F_grid": 0.0,
            "down_speed": 0.0,
            "X1": float(np.sum((q / np.maximum(gd, 1e-12)) * vol)),
            "X2": float(np.sum(q * vol)),
            "P_grav": float(np.sum(density * (-G_MAG) * v[:, 2] * vol)),
            "KE": ke,
            "max_speed": float(np.max(np.linalg.norm(v, axis=1))),
        })
        # no early stop: the post-splat creep tail carries the low-rate rows that
        # pin tau_y, so every drop observes the same full window
    meta = {"v_impact": v_impact, "mass_kg": mass, "n_particles": int(len(pos)),
            "ke_impact_j": 0.5 * mass * v_impact * v_impact + mass * G_MAG * BOTTOM_GAP}
    return records, meta


def evaluate_drop(
    records: list[dict],
    meta: dict,
    height: float,
    sigma_power: float,
    theta_ref: tuple[float, float],
    seed: int,
) -> dict:
    """Fit one drop with the press sweep's estimator and mirror its summary fields."""
    active = [r for r in records
              if r["tick"] > 0 and np.isfinite(r["X2"]) and r["X2"] > 1e-14]
    fit = pcs._fit_power_balance(active, sigma_power=sigma_power, theta_ref=theta_ref)
    tau_y, eta = theta_ref
    summary = {
        "seed": seed,
        "fit_method": "power_balance",
        "height_m": height,
        "v_impact": meta["v_impact"],
        "ke_impact_j": meta["ke_impact_j"],
        "mass_kg": meta["mass_kg"],
        "n_particles": meta["n_particles"],
        "ticks_run": len(records) - 1,
        "active_rows": len(active),
        "final_ke_j": records[-1]["KE"],
        "final_max_speed": records[-1]["max_speed"],
    }
    if fit["ok"]:
        fixed, fitn = fit["fixed_noise"], fit["fit_noise"]
        summary.update({
            "tau_y_hat": float(fit["theta"][0]),
            "eta_hat": float(fit["theta"][1]),
            "tau_y_err": abs(float(fit["theta"][0]) / tau_y - 1.0),
            "eta_err": abs(float(fit["theta"][1]) / eta - 1.0),
            "law_err": pcs._law_error(fit["theta"], theta_ref, fixed["AtA"]),
            "law_err_std": float(np.sqrt(2.0) * sigma_power
                                 / max(np.sqrt(np.asarray(theta_ref) @ fixed["AtA"]
                                               @ np.asarray(theta_ref)), 1e-30)),
            **pcs._noise_recovery_rms(fit, theta_ref, sigma_power,
                                      seed=seed * 7919 + int(round(height * 1e6))),
            "fit_relres": fit["fit_relres"],
            "sigma_fit_w": fit["sigma_fit"],
            "rank": fixed["rank"],
            "cond": fixed["cond"],
            "std_tau_y_fixed": fixed["std_tau_y"],
            "std_eta_fixed": fixed["std_eta"],
            "rel_std_tau_y_fixed": fixed["rel_std_tau_y"],
            "rel_std_eta_fixed": fixed["rel_std_eta"],
            "corr_tau_y_eta_fixed": fixed["corr_tau_y_eta"],
            "major_axis_std_fixed": fixed["major_axis_std"],
            "std_tau_y_fit": fitn["std_tau_y"],
            "std_eta_fit": fitn["std_eta"],
            "rel_std_tau_y_fit": fitn["rel_std_tau_y"],
            "rel_std_eta_fit": fitn["rel_std_eta"],
            "cov_fixed": fixed["cov"].tolist(),
            "AtA": fixed["AtA"].tolist(),
            "eig_info": fixed["eig_info"].tolist(),
        })
    else:
        summary.update({**pcs.FIT_FAIL_FIELDS, "reason": fit["reason"]})
    return {"summary": summary, "records": records}


def write_outputs(
    results: list[dict],
    out_dir: Path,
    *,
    tau_y: float,
    eta: float,
    heights: list[float],
    seeds: list[int],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*_records.npz", "*.csv", "*.json"):
        for old in out_dir.glob(pattern):
            old.unlink()

    summaries = [r["summary"] for r in results]
    _key = pcs._target_key
    by_seed_height = {
        seed: {_key(r["summary"]["height_m"]): r["summary"] for r in results
               if r["summary"]["seed"] == seed}
        for seed in seeds
    }
    pcs._write_rows(out_dir, "results", summaries,
                    fieldnames=pcs._ordered_summary_keys(summaries))
    for r in results:
        s = r["summary"]
        recs = r["records"]
        np.savez_compressed(
            out_dir / f"seed_{s['seed']}_h{round(s['height_m'] * 1000):04d}mm_records.npz",
            data=np.array([[rec[k] for k in recs[0]] for rec in recs], dtype=float),
            keys=np.array(list(recs[0].keys())),
        )

    def band(key: str):
        return pcs._finite_percentiles(
            [[by_seed_height[s][_key(h)][key] for h in heights] for s in seeds])

    H = np.array(sorted(heights))
    heights = sorted(heights)

    def band_plot(filename, title, ylabel, curves, log_y=False):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for x, (p25, med, p75), color, style, label in curves:
            ok = np.isfinite(x) & np.isfinite(med)
            ax.fill_between(x[ok], p25[ok], p75[ok], color=color, alpha=0.16, linewidth=0)
            ax.plot(x[ok], med[ok], style, lw=2.1, color=color, label=label)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel("drop height  (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    rel_tau = band("rel_std_tau_y_fixed")
    rel_eta = band("rel_std_eta_fixed")
    drop_rows = [{
        "height_m": h,
        "v_impact": by_seed_height[seeds[0]][_key(h)]["v_impact"],
        "n_seeds": len(seeds),
        "rel_std_tau_y": rel_tau[1][i],
        "rel_std_tau_y_p25": rel_tau[0][i],
        "rel_std_tau_y_p75": rel_tau[2][i],
    } for i, h in enumerate(heights)]
    pcs._write_rows(out_dir, "drop_results", drop_rows)

    band_plot("relative_uncertainty_vs_height.png", "Yield Stress Uncertainty",
              r"yield-stress uncertainty  $\sqrt{C_{\tau_y\tau_y}} / \tau_y$",
              [(H, rel_tau, "tab:blue", "o-", "median across seeds")], log_y=True)
    band_plot("viscosity_uncertainty_vs_height.png", "Viscosity Uncertainty",
              r"viscosity uncertainty  $\sqrt{C_{\eta\eta}} / \eta$",
              [(H, rel_eta, "tab:orange", "o-", "median across seeds")], log_y=True)

    resb = band("fit_relres")
    res_med = np.where(resb[1] > 1e-6, resb[1], np.nan)
    band_plot("law_error_vs_height.png", "Identified-Law Error over the Probed Rates",
              r"relative law error  $\|A(\hat\theta - \theta^*)\|\,/\,\|A\theta^*\|$",
              [(H, band("law_err"), "tab:blue", "o-", "identified law vs truth"),
               (H, (res_med, res_med, res_med), "0.45", ":", "power-prediction residual")],
              log_y=True)

    def abs_pct(series, true_value, rank_med):
        med = np.maximum(np.abs(series[1] / true_value - 1.0) * 100.0, 1e-2)
        med = np.where(rank_med >= 2, med, np.nan)
        return (med, med, med)

    rank_med = band("rank")[1]
    band_plot("parameter_error_vs_height.png", "Parameter Recovery Error",
              "absolute parameter error  (%)",
              [(H, abs_pct(band("tau_y_hat"), tau_y, rank_med), "tab:blue", "o-", r"$\tau_y$"),
               (H, abs_pct(band("eta_hat"), eta, rank_med), "tab:orange", "s-", r"$\eta$")],
              log_y=True)

    print(f"\nwrote drop sweep outputs to {out_dir}")


def run(args) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed is not None else list(dict.fromkeys(int(s) for s in args.seeds))
    theta_ref = (args.tau_y, args.eta)
    heights, seen = [], set()
    for h in args.heights:
        key = pcs._target_key(h)
        if key not in seen:
            heights.append(float(h))
            seen.add(key)

    results = []
    for seed in seeds:
        for height in sorted(heights):
            records, meta = run_drop(
                seed, height, device=args.device, n_grid=args.n_grid, ticks=args.ticks,
                substeps=args.substeps, dt=args.dt, radius=args.radius, ppc=args.ppc,
                tau_y=args.tau_y, eta=args.eta, density=args.density, bulk=args.bulk,
            )
            result = evaluate_drop(records, meta, height, args.sigma_power, theta_ref, seed)
            s = result["summary"]
            print(
                f"seed {seed}  h={height:6.3f} m  v0={s['v_impact']:.2f} m/s  "
                f"ticks={s['ticks_run']}, rows={s['active_rows']}, "
                f"theta=({s['tau_y_hat']:.1f}, {s['eta_hat']:.1f}), "
                f"rel_std=({s['rel_std_tau_y_fixed']:.3g}, {s['rel_std_eta_fixed']:.3g}), "
                f"law_err={s['law_err']:.3g}",
                flush=True,
            )
            results.append(result)

    write_outputs(results, OUT, tau_y=args.tau_y, eta=args.eta,
                  heights=sorted(heights), seeds=seeds)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep ball drop height and plot parameter covariance.")
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    parser.add_argument("--heights", nargs="+", type=float,
                        default=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                        help="Equivalent drop heights in m, log-spaced 2 mm - 1 m. The bottom "
                             "is the no-information plop (yielding needs tau_y/(2 rho g) ~ 1 cm); "
                             "0.5-1.0 m deliberately enter the overconfidence regime where the "
                             "energy route breaks (rho v^2 >> 20 tau_y): tau_y_hat inflates to "
                             "650-2300 Pa while its sigma shrinks to 2-4%.")
    parser.add_argument("--n-grid", type=int, default=128,
                        help="Grid cells per side (3.1 mm cells: ~19 across the default ball).")
    parser.add_argument("--ticks", type=int, default=150,
                        help="Control ticks (2 ms each): a fixed 0.3 s observation window "
                             "covering crush, spread, and the tau_y-informative creep tail.")
    parser.add_argument("--substeps", type=int, default=80)
    parser.add_argument("--dt", type=float, default=2.5e-5)
    parser.add_argument("--radius", type=float, default=0.03, help="Ball radius in m.")
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--sigma-power", type=float, default=0.05,
                        help="Assumed power-noise (W) for the fixed-noise covariance.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Sampling-jitter seeds; each runs the full height ladder.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (overrides --seeds).")
    parser.add_argument("--tau-y", type=float, default=200.0)
    parser.add_argument("--eta", type=float, default=40.0)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--bulk", type=float, default=9.0e5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
