"""Probe design for the dough press: which action determines which constant.

The paper's design rule (Sec. 3, Sec. 6) says the information matrix A^T A of a
probe is known from its motion alone, before any law is fit, so the agent can
CHOOSE the action that excites the constants its task needs. This example makes
that concrete for the plate press on Bingham dough, theta = (tau_y, eta), with
two probe families:

1. SPEED sweep (velocity-controlled): the same 30 mm press at different plate
   speeds. Speed moves the shear-rate window (rate ~ speed / gap), and the rate
   window decides identifiability: slow presses pin the yield stress and leave
   viscosity loose, fast presses pin the high-rate combination tau_y + eta*gd,
   and only pooling different speeds pins both. Each probe gets its predicted
   2-sigma covariance ellipse sigma^2 (A^T A)^-1 overlaid with a Monte-Carlo
   recovery cloud (noise in the measured power b, plus 1% relative noise on the
   motion integrals in A — so the cloud tests the ellipse as the calibrated
   approximation the paper says it is, not as a tautology).

2. FORCE-controlled probe (admittance): command a target contact force and let
   the material decide. Below the load the dough can hold statically, nothing
   flows -> the weak form collects no informative rows and tau_y is returned as
   UNIDENTIFIED (rank-deficient information); just above, flow starts and
   identification switches on. "Too small a pressure cannot read the yield
   stress" is true exactly here — in force control — and the information
   matrix flags it before any fit.

Noise model: sigma_power (W) i.i.d. on each power-balance row, the same fixed
assumption as pressure_covariance_sweep. The analytic ellipse uses b-noise
only; the MC cloud adds kinematic noise, mirroring the paper's caveat that
tracking noise also perturbs A.

Run (paper-quality, ~15-25 min on one GPU):
  python examples/probe_design_covariance.py --device cuda:1
Quick biased iteration mode (~1 min):
  python examples/probe_design_covariance.py --device cuda:1 --n-grid 48 --dt 1e-4 --substeps 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pressure_covariance_sweep as pcs  # noqa: E402  (shared press + fit machinery)

from warpmpm import GridConfig, Solver, block, newtonian  # noqa: E402
from warpmpm.coupling.backend import WarpMPMBackend  # noqa: E402
from warpmpm.coupling.admittance import ForceAdmittance  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out" / "probe_design"
THETA_TRUE = (200.0, 40.0)
G_MAG = 9.81
# fixed short engagement protocol: speed probes are as short as ~47 ticks at the
# fastest speed, so they cannot afford the pressure sweep's long approach ramp
RAMP_TICKS = 10
ENGAGE_TICKS = 12


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

def active_rows(records: list[dict], f_min: float = 0.25,
                v_min: float = 2e-4, x2_min: float = 1e-6) -> list[dict]:
    """Genuinely flowing, in-contact rows of a recorded press.

    v_min (m/s) and x2_min reject the numerical dust a regulated-but-static
    plate produces (grid contact-force quantization -> micro-velocity jitter),
    which would otherwise read as hundreds of fake observations. The first
    ENGAGE_TICKS are excluded like in the pressure sweep: engagement-transient
    rows are contaminated (and the artifact grows with grid resolution)."""
    return [r for r in records
            if r["tick"] > ENGAGE_TICKS and r["down_speed"] > v_min
            and np.isfinite(r["X2"]) and r["X2"] > x2_min and r["F_grid"] > f_min]


def run_speed_probe(v: float, args) -> dict:
    """One velocity-controlled press of fixed travel at plate speed v."""
    dt_ctrl = args.dt * args.substeps
    ticks = max(int(round(args.travel / (v * dt_ctrl))), 8)
    records, _ = pcs.run_press(
        args.seed, device=args.device, n_grid=args.n_grid, ticks=ticks,
        substeps=args.substeps, dt=args.dt, v_max=v,
        tau_y=THETA_TRUE[0], eta=THETA_TRUE[1], density=1000.0, bulk=9e5,
        dough_size=(0.12, 0.12, 0.06), ramp_ticks=RAMP_TICKS,
    )
    rows = active_rows(records)
    fit = pcs._fit_power_balance(rows, sigma_power=args.sigma_power, theta_ref=THETA_TRUE)
    gd_c = [r["X2"] / r["X1"] for r in rows if r["X1"] > 1e-12]
    return {"v": v, "ticks": ticks, "rows": len(rows), "fit": fit,
            "rate_lo": float(np.percentile(gd_c, 10)) if gd_c else float("nan"),
            "rate_hi": float(np.percentile(gd_c, 90)) if gd_c else float("nan")}


def run_force_probe(f_target: float, args) -> dict:
    """One force-controlled press: admittance regulates the plate to f_target.

    Below the static holding load the plate settles almost immediately (no flow,
    no informative rows); above it the dough creeps and rows accumulate.
    """
    grid = GridConfig(n_grid=args.n_grid, grid_lim=0.4)
    pos, vol0, floor = block(grid, size=(0.12, 0.12, 0.06), ppc=2, seed=args.seed)
    s = Solver(grid=grid, device=args.device).load_particles(pos, vol0)
    s.set_material(newtonian(eta=THETA_TRUE[1], density=1000.0, bulk_modulus=9e5)
                   .with_yield(THETA_TRUE[0]))
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    cx = cy = 0.2
    half = (0.075, 0.075, 0.6 * grid.dx)
    dt_ctrl = args.dt * args.substeps
    backend = WarpMPMBackend(solver=s)
    z0 = floor + 0.06 + half[2]
    z = z0
    z_floor = floor + half[2] + 0.003
    tool = backend.attach_tool((cx, cy, z), half, velocity=(0, 0, 0))
    # stiff damping + a low-pass on the measured force keep the controller from
    # chasing the grid contact-force quantization (~0.5 N steps)
    adm = ForceAdmittance(f_target=f_target, v_max=0.08, allow_retract=True,
                          damping=max(150.0, f_target / 0.08))

    records, z_hist, f_ema, regulated_at = [], [z], 0.0, None
    for tick in range(args.force_ticks + 1):
        v_down = adm.velocity_down(f_ema) if tick > 0 else 0.0
        z_new = float(np.clip(z - v_down * dt_ctrl, z_floor, z0))
        vz = (z_new - z) / dt_ctrl
        if tick > 0:
            backend.set_tool_kinematics(tool, center=(cx, cy, z), velocity=(0, 0, vz))
            backend.reset_tool_force(tool)
            backend.step(args.dt, args.substeps)
            z = z_new
        # feedback: the compressive-gated static stress wrench, low-passed. It
        # is UNCALIBRATED — it carries an O(1 N) self-weight offset and under-
        # reads during fast flow — i.e., an untared wrist sensor. The offset
        # merely shifts the effective setpoint; the information switch this
        # probe demonstrates does not depend on sensor calibration. (The
        # Newton-exact grid impulse is NOT usable for control here: it
        # collapses at a halt and reads tensile artifacts at creep speeds.)
        f = max(float(backend.get_tool_reaction(tool, dt_ctrl)[2]), 0.0) if tick > 0 else 0.0
        f_static = backend.get_tool_wrench(tool, at_center=(cx, cy, z))["Fz"] if tick > 0 else 0.0
        f_ema = f_static if tick <= 1 else 0.8 * f_ema + 0.2 * f_static
        if regulated_at is None and tick > 0 and f_ema >= 0.9 * f_target:
            regulated_at = tick
        vp, L, vol = s.v(), s.L(), s.vol()
        gd, q = pcs.equivalent_shear_rate(L)
        records.append({
            "tick": tick, "t": tick * dt_ctrl,
            "down_speed": max(0.0, -vz) if tick > 0 else 0.0,
            "F_grid": f,
            "F_fb": f_ema if tick > 0 else 0.0,
            "X1": float(np.sum((q / np.maximum(gd, 1e-12)) * vol)),
            "X2": float(np.sum(q * vol)),
            "P_grav": float(np.sum(1000.0 * (-G_MAG) * vp[:, 2] * vol)),
            "KE": float(0.5 * 1000.0 * np.sum(vol * np.sum(vp * vp, axis=1))),
        })
        z_hist.append(z)
        # settled = plate effectively stationary over the last 50 control ticks
        if tick > 60 and (z_hist[-51] - z) < 2e-5:
            break

    # the yield question is answered by rows where the material FLOWS while the
    # feedback force is near the command: a command the dough can hold statically
    # only flows on the approach (excluded), a command beyond the holding load
    # flows indefinitely at the target (included) — the approach transient
    # reflects the controller, not the material's limit load
    near_target = [r for r in records if r["F_fb"] >= 0.85 * f_target]
    rows = active_rows(near_target)
    fit = pcs._fit_power_balance(rows, sigma_power=args.sigma_power, theta_ref=THETA_TRUE)
    settled = len(records) - 1 < args.force_ticks
    depth_mm = (z0 - z) * 1e3
    return {"f_target": f_target, "rows": len(rows), "fit": fit,
            "settled": settled, "ticks_run": len(records) - 1,
            "regulated_at": regulated_at, "final_depth_mm": depth_mm, "f_end": f}


# ---------------------------------------------------------------------------
# covariance ellipse + Monte-Carlo recovery cloud
# ---------------------------------------------------------------------------

def mc_cloud(A: np.ndarray, b: np.ndarray, sigma_power: float,
             n: int = 800, a_rel: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2))
    for i in range(n):
        An = A * (1.0 + rng.normal(0.0, a_rel, size=A.shape))
        bn = b + rng.normal(0.0, sigma_power, size=b.shape)
        out[i] = np.linalg.lstsq(An, bn, rcond=None)[0]
    return out


def ellipse_points(cov: np.ndarray, center: np.ndarray, nsig: float = 2.0, n: int = 128):
    w, V = np.linalg.eigh(cov)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    return center[:, None] + nsig * (V @ (np.sqrt(np.maximum(w, 0.0))[:, None] * circ))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(speed_results: list[dict], force_results: list[dict], args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    ok = [r for r in speed_results if r["fit"]["ok"]]

    # --- figure A: ellipse-vs-scatter panel per speed + pooled ---------------
    n_pan = len(ok) + 1
    ncols = (n_pan + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(4.0 * ncols, 7.6))
    axes = np.atleast_2d(axes).ravel()
    pooled_A = np.vstack([r["fit"]["A"] for r in ok])
    pooled_b = np.concatenate([r["fit"]["b"] for r in ok])
    pooled_fit = pcs._fit_design_matrix(pooled_A, pooled_b,
                                        sigma_power=args.sigma_power, theta_ref=THETA_TRUE)
    panels = [(f"v = {r['v']*1000:.0f} mm/s", r["fit"]) for r in ok]
    panels.append(("all speeds pooled", pooled_fit))
    for ax, (title, fit) in zip(axes, panels):
        th = np.asarray(fit["theta"], dtype=float)
        cloud = mc_cloud(fit["A"], fit["b"], args.sigma_power)
        stats = fit["fixed_noise"]
        ell = ellipse_points(stats["cov"], th)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=3, alpha=0.2, color="tab:blue",
                   label="recovery cloud (noise MC)")
        ax.plot(ell[0], ell[1], color="tab:orange", lw=2.0, label=r"predicted $2\sigma$ ellipse")
        ax.plot(*THETA_TRUE, marker="*", ms=14, color="tab:red", ls="none", label="truth")
        ax.plot(th[0], th[1], marker="o", ms=5, color="black", ls="none", label=r"$\hat\theta$")
        ax.set_title(f"{title}\n"
                     rf"$\sigma(\tau_y)/\tau_y$={stats['rel_std_tau_y']:.1%}, "
                     rf"$\sigma(\eta)/\eta$={stats['rel_std_eta']:.1%}", fontsize=10)
        ax.set_xlabel(r"$\tau_y$ (Pa)")
        ax.set_ylabel(r"$\eta$ (Pa s)")
        ax.grid(alpha=0.3)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("What each press speed pins down: predicted covariance vs recovery scatter",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "speed_ellipses.png", dpi=150)
    plt.close(fig)

    # --- figure B: design curves --------------------------------------------
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    V = np.array([r["v"] for r in ok])
    rel_ty = np.array([r["fit"]["fixed_noise"]["rel_std_tau_y"] for r in ok])
    rel_et = np.array([r["fit"]["fixed_noise"]["rel_std_eta"] for r in ok])
    ax.loglog(V, rel_ty, "o-", color="tab:blue", lw=2, label=r"$\sigma(\tau_y)/\tau_y$")
    ax.loglog(V, rel_et, "s-", color="tab:orange", lw=2, label=r"$\sigma(\eta)/\eta$")
    ps = pooled_fit["fixed_noise"]
    ax.axhline(ps["rel_std_tau_y"], color="tab:blue", ls=":", lw=1.5,
               label=r"pooled $\sigma(\tau_y)/\tau_y$")
    ax.axhline(ps["rel_std_eta"], color="tab:orange", ls=":", lw=1.5,
               label=r"pooled $\sigma(\eta)/\eta$")
    ax.set_xlabel("plate speed (m/s)")
    ax.set_ylabel("predicted relative uncertainty")
    ax.set_title("absolute uncertainty (fixed power noise:\nfaster press = larger signal)",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    axr.semilogx(V, rel_ty / rel_et, "d-", color="tab:green", lw=2)
    axr.axhline(1.0, color="0.4", ls="--", lw=1.2)
    axr.text(V[0], 1.05, r"$\tau_y$-favored below, $\eta$-favored above", fontsize=9, va="bottom")
    axr.set_xlabel("plate speed (m/s)")
    axr.set_ylabel(r"$\sigma(\tau_y)/\tau_y \;/\; \sigma(\eta)/\eta$")
    axr.set_title("which parameter the probe favors\nflips with speed — the design signal",
                  fontsize=11)
    axr.grid(alpha=0.3, which="both")
    fig.suptitle("The design rule for the press: speed selects the shear-rate window, "
                 "the window selects the parameter", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "speed_design_curves.png", dpi=150)
    plt.close(fig)

    # --- figure C: force-controlled information switch ----------------------
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1.0, 1.3]})
    F = np.array([r["f_target"] for r in force_results])
    rows = np.array([r["rows"] for r in force_results])
    a0.bar(np.arange(len(F)), rows, color=["0.6" if r["rows"] < 4 else "tab:green"
                                           for r in force_results])
    a0.set_ylabel("informative rows")
    a0.set_title("Force-controlled probe: below the holding load nothing flows,\n"
                 "so the yield stress is UNIDENTIFIED — and the information matrix says so")
    a0.grid(alpha=0.3, axis="y")
    rel = []
    for r in force_results:
        fit = r["fit"]
        rel.append(fit["fixed_noise"]["rel_std_tau_y"] if fit.get("ok") else np.inf)
    rel = np.array(rel)
    xpos = np.arange(len(F))
    fin = np.isfinite(rel)
    a1.semilogy(xpos[fin], rel[fin], "o-", color="tab:blue", lw=2)
    if (~fin).any():
        cap = (np.nanmin(rel[fin]) * 1e4) if fin.any() else 1.0
        a1.semilogy(xpos[~fin], np.full((~fin).sum(), cap), marker="x", ms=11,
                    color="tab:red", ls="none", label="rank-deficient: unidentified")
        a1.legend(fontsize=9)
    a1.set_xticks(xpos, [f"{f:.0f}" for f in F])
    a1.set_xlabel("commanded contact force (N)")
    a1.set_ylabel(r"predicted $\sigma(\tau_y)/\tau_y$")
    a1.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "force_probe_information.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    speed_results = []
    for v in args.speeds:
        print(f"\n=== speed probe v = {v*1000:.0f} mm/s ===", flush=True)
        r = run_speed_probe(v, args)
        s = r["fit"]["fixed_noise"] if r["fit"]["ok"] else None
        print(f"rows={r['rows']} rates {r['rate_lo']:.2f}-{r['rate_hi']:.2f}/s " +
              (f"theta=({r['fit']['theta'][0]:.1f}, {r['fit']['theta'][1]:.1f}) "
               f"rel_std=({s['rel_std_tau_y']:.3g}, {s['rel_std_eta']:.3g})" if s else "fit failed"),
              flush=True)
        speed_results.append(r)

    force_results = []
    for f_target in args.force_targets:
        print(f"\n=== force probe target = {f_target:.0f} N ===", flush=True)
        r = run_force_probe(f_target, args)
        ident = r["fit"].get("ok", False)
        print(f"settled={r['settled']} depth={r['final_depth_mm']:.1f}mm rows={r['rows']} "
              f"identified={ident}" +
              (f" theta=({r['fit']['theta'][0]:.1f}, {r['fit']['theta'][1]:.1f})" if ident else ""),
              flush=True)
        force_results.append(r)

    make_figures(speed_results, force_results, args)

    def strip(r):
        out = {k: v for k, v in r.items() if k != "fit"}
        if r["fit"].get("ok"):
            st = r["fit"]["fixed_noise"]
            out.update({"tau_y_hat": float(r["fit"]["theta"][0]),
                        "eta_hat": float(r["fit"]["theta"][1]),
                        "rel_std_tau_y": st["rel_std_tau_y"],
                        "rel_std_eta": st["rel_std_eta"],
                        "cond": st["cond"],
                        "law_err": pcs._law_error(r["fit"]["theta"], THETA_TRUE, st["AtA"])})
        else:
            out.update({"tau_y_hat": None, "eta_hat": None,
                        "rel_std_tau_y": None, "rel_std_eta": None, "cond": None,
                        "law_err": None, "reason": r["fit"].get("reason")})
        return out

    (OUT / "speed_results.json").write_text(
        json.dumps([strip(r) for r in speed_results], indent=2, default=float))
    (OUT / "force_results.json").write_text(
        json.dumps([strip(r) for r in force_results], indent=2, default=float))
    print(f"\nwrote {OUT}/speed_ellipses.png, speed_design_curves.png, "
          f"force_probe_information.png (+ json)")


def parse_args():
    p = argparse.ArgumentParser(description="Probe design via the information matrix for the dough press.")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-grid", type=int, default=128)
    p.add_argument("--dt", type=float, default=2.5e-5)
    p.add_argument("--substeps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--travel", type=float, default=0.030, help="Press travel (m) for speed probes.")
    p.add_argument("--speeds", nargs="+", type=float,
                   default=[0.02, 0.04, 0.08, 0.16, 0.32], help="Plate speeds (m/s).")
    p.add_argument("--force-targets", nargs="+", type=float,
                   default=[2.0, 4.0, 8.0, 16.0, 24.0, 32.0],
                   help="Commanded contact forces (N, uncalibrated sensor) for the admittance "
                        "probe; bracket the dough's initial static holding load (~11 N).")
    p.add_argument("--force-ticks", type=int, default=800,
                   help="Max control ticks per force probe.")
    p.add_argument("--sigma-power", type=float, default=0.05)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
