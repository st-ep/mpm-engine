"""Force/pressure sweep for viscoplastic dough-press identifiability.

Headless experiment for the paper's covariance claim. The plate motion does not
depend on the stop force, so ONE constant-speed press is simulated per seed and
recorded per control tick; every target load is then evaluated as a prefix of
that trajectory, cut at the first tick whose grid reaction force reaches the
target. The measured motion builds the weak-form power-balance design matrix
for theta = [tau_y, eta]:

    P_dev = tau_y * INT(q / gamma_dot) dV + eta * INT(q) dV,
    q = 2 dev(D):dev(D), gamma_dot = sqrt(q + eps^2).

For each target we report the recovered parameters, the information matrix
A^T A, and cov(theta_hat) = sigma_power^2 (A^T A)^-1 under a fixed power-noise
assumption. Plots put the REALIZED plate pressure (the force actually reached)
on the x-axis, so an unreachable target can never mislabel an axis. Recovery is
plotted as the LAW error ||A(theta_hat - theta*)|| / ||A theta*|| — the relative
flow-curve error weighted by the rates the press actually visited — because the
raw coordinates (tau_y, eta) are near-degenerate along tau_y + eta*gd ~ const;
the signed coordinate errors are shown separately in percent.

Engagement: the plate creeps into contact (V_CREEP for CREEP_TICKS), then
accelerates slowly to v_max over ACCEL_TICKS. Plate force is strain-rate
dominated, so the acceleration sweeps the force through the 0.3-0.5 kPa band
tick by tick; a full-speed engagement instead lands its first clean tick at
~0.5 kPa and every smaller stop-force target gets zero usable rows at 128^3.

Valid regime: the defaults (ticks=280, targets 0.3-2 kPa on the default
dough) keep the plate-floor gap at or above ~2 grid cells. Pressing deeper
leaves the model's validity — the squeeze film is unresolved and the EOS
compression power (absent from the two-term dissipation model) grows — which
biases theta while the fixed-noise covariance keeps shrinking.

Resolution: the default grid is 128^3 (3.1 mm cells; dt=2.5e-5 x 80 substeps
keeps the same 2 ms control tick). At this resolution the MPM transfer/BC
dissipation absorbed by the fit is small enough that the identified law lands
within ~6-15% of truth; at 48^3 the same absorbed dissipation inflates eta by
up to 2x (law error 19-40%). Quick iteration mode:
  --n-grid 48 --dt 1e-4 --substeps 20   (~20x faster, biased)

Important caveat: this is the scalar power-balance diagnostic, not the paper's
full divergence-free tensor weak-form estimator. The paper-style estimator for
the press needs discrete virtual work of the contact load, i.e. contact/grid
impulse distribution weighted by the same divergence-free test fields used in
A. A total plate force is enough for scalar power, but not enough for
arbitrary tensor weak-form rows.

Run:
  python experiments/pressure_covariance_sweep.py --device cuda:0
  python experiments/pressure_covariance_sweep.py --device cuda:0 --seed 0  # quick single-seed run
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.coupling.backend import WarpMPMBackend

OUT = Path(__file__).resolve().parents[1] / "out" / "pressure_covariance_sweep"
G_MAG = 9.81
EPS_GAMMA = 0.05
RAMP_TICKS = 30      # legacy linear approach ramp (used when creep engagement is off)
ENGAGE_TICKS = 8     # first-touch transient rows are excluded from every fit
# Gentle-touch engagement. Plate force on this dough is strain-RATE dominated,
# not depth dominated: at V_CREEP it saturates near 2-5 N (0.1-0.2 kPa) while a
# full-speed engagement's first clean tick already reads ~11 N (0.5 kPa). The
# 0.3-0.45 kPa stop forces exist only at intermediate speeds, so after a short
# creep the plate accelerates SLOWLY: the force sweeps ~5 -> 13 N over the
# acceleration and every small target collects a distinct crossing with rows.
V_CREEP = 0.02      # m/s; touch speed (force ~2-3 N once engaged)
CREEP_TICKS = 60    # creep duration before the slow acceleration starts
ACCEL_TICKS = 80    # creep -> v_max over this many ticks (~0.11 N per tick)

_NAN = float("nan")
_INF = float("inf")
FIT_FAIL_FIELDS = {
    "tau_y_hat": _NAN, "eta_hat": _NAN, "tau_y_err": _NAN, "eta_err": _NAN,
    "law_err": _NAN, "law_err_std": _NAN,
    "tau_y_err_rms": _NAN, "eta_err_rms": _NAN, "law_err_rms": _NAN,
    "fit_relres": _NAN, "sigma_fit_w": _NAN, "rank": 0, "cond": _INF,
    "std_tau_y_fixed": _INF, "std_eta_fixed": _INF,
    "rel_std_tau_y_fixed": _INF, "rel_std_eta_fixed": _INF,
    "corr_tau_y_eta_fixed": _NAN, "major_axis_std_fixed": _INF,
    "std_tau_y_fit": _INF, "std_eta_fit": _INF,
    "rel_std_tau_y_fit": _INF, "rel_std_eta_fit": _INF,
    "cov_fixed": [[_NAN, _NAN], [_NAN, _NAN]],
    "AtA": [[_NAN, _NAN], [_NAN, _NAN]],
    "eig_info": [_NAN, _NAN],
}


def equivalent_shear_rate(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (gamma_dot_eps, q), q = 2 dev(D):dev(D), D = sym(L)."""
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    tr = (D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2]) / 3.0
    dev = D - tr[..., None, None] * np.eye(3)
    q = 2.0 * np.einsum("...ij,...ij->...", dev, dev)
    return np.sqrt(q + EPS_GAMMA * EPS_GAMMA), q


def _noise_recovery_rms(fit: dict, theta_ref: tuple[float, float], sigma_power: float,
                        n_draws: int = 256, seed: int = 0) -> dict:
    """Expected recovery error under the measurement-noise model the covariance
    assumes: refit with i.i.d. N(0, sigma_power) added to each power reading and
    report RMS relative errors over the draws. This is what a real (noisy)
    experiment would return — the deterministic simulation alone never samples
    the weakly-determined direction, so its single draw understates the error
    of an insufficient probe."""
    A, b = fit["A"], fit["b"]
    tt = np.asarray(theta_ref, dtype=float)
    AtA = fit["fixed_noise"]["AtA"]
    rng = np.random.default_rng(seed)
    ths = np.empty((n_draws, 2))
    for i in range(n_draws):
        ths[i] = np.linalg.lstsq(A, b + rng.normal(0.0, sigma_power, size=b.shape),
                                 rcond=None)[0]
    rel = ths / tt - 1.0
    d = ths - tt
    law = np.sqrt(np.maximum(np.einsum("ki,ij,kj->k", d, AtA, d), 0.0)
                  / max(tt @ AtA @ tt, 1e-30))
    return {
        "tau_y_err_rms": float(np.sqrt(np.mean(rel[:, 0] ** 2))),
        "eta_err_rms": float(np.sqrt(np.mean(rel[:, 1] ** 2))),
        "law_err_rms": float(np.sqrt(np.mean(law ** 2))),
    }


def _law_error(theta_hat: np.ndarray, theta_ref: tuple[float, float], AtA: np.ndarray) -> float:
    """Relative flow-curve error in the probe's own metric, ||A (th-t*)|| / ||A t*||:
    the dissipation-weighted mismatch of the identified law over the rates the
    press actually visited. Insensitive to the near-null trade-off direction."""
    d = np.asarray(theta_hat, dtype=float) - np.asarray(theta_ref, dtype=float)
    tt = np.asarray(theta_ref, dtype=float)
    return float(np.sqrt(max(d @ AtA @ d, 0.0) / max(tt @ AtA @ tt, 1e-30)))


def _information_stats(A: np.ndarray, sigma_power: float, theta_ref: tuple[float, float]) -> dict:
    AtA = A.T @ A
    eig_info = np.linalg.eigvalsh(AtA)
    rank = int(np.linalg.matrix_rank(AtA, tol=max(float(eig_info[-1]), 1.0) * 1e-10))
    if rank == 2:
        cond = float(np.linalg.cond(AtA))
        cov = sigma_power * sigma_power * np.linalg.inv(AtA)
        eig_cov = np.linalg.eigvalsh(cov)
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        corr = float(cov[0, 1] / max(std[0] * std[1], 1e-30))
        major_std = float(np.sqrt(max(eig_cov[-1], 0.0)))
    else:
        cond = _INF
        cov = np.full((2, 2), np.nan)
        std = np.array([_INF, _INF])
        corr = _NAN
        major_std = _INF
    tau_y, eta = theta_ref
    return {
        "AtA": AtA, "eig_info": eig_info, "rank": rank, "cond": cond, "cov": cov,
        "std_tau_y": float(std[0]), "std_eta": float(std[1]),
        "rel_std_tau_y": float(std[0] / max(abs(tau_y), 1e-30)),
        "rel_std_eta": float(std[1] / max(abs(eta), 1e-30)),
        "corr_tau_y_eta": corr,
        "major_axis_std": major_std,
    }


def _fit_design_matrix(A: np.ndarray, b: np.ndarray, sigma_power: float,
                       theta_ref: tuple[float, float]) -> dict:
    # even a 1-row system returns its (minimum-norm) point: an insufficient
    # probe should REPORT its blown-up error and covariance, not go silent
    if len(b) < 1:
        return {"ok": False, "reason": "no rows"}
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ theta
    dof = max(len(b) - 2, 1)
    sigma_fit = float(np.sqrt(np.dot(resid, resid) / dof))
    return {
        "ok": True, "A": A, "b": b, "theta": theta,
        "sigma_fit": sigma_fit,
        "fit_relres": float(np.linalg.norm(resid) / max(np.linalg.norm(b), 1e-30)),
        "n_rows": int(len(b)),
        "fixed_noise": _information_stats(A, sigma_power, theta_ref),
        "fit_noise": _information_stats(A, sigma_fit, theta_ref),
    }


def _fit_power_balance(records: list[dict], sigma_power: float,
                       theta_ref: tuple[float, float]) -> dict:
    if len(records) < 1:
        return {"ok": False, "reason": "no active rows"}
    t = np.array([r["t"] for r in records])
    KE = np.array([r["KE"] for r in records])
    A_rows, b_rows = [], []
    for i, r in enumerate(records):
        i0, i1 = max(i - 1, 0), min(i + 1, len(records) - 1)
        dKE = (KE[i1] - KE[i0]) / max(t[i1] - t[i0], 1e-30)
        if r["X1"] <= 1e-14 or r["X2"] <= 1e-14:
            continue
        A_rows.append((r["X1"], r["X2"]))
        b_rows.append(r["F_grid"] * r["down_speed"] + r["P_grav"] - dKE)
    if len(A_rows) < 1:
        return {"ok": False, "reason": "no nonzero rows"}
    return _fit_design_matrix(np.asarray(A_rows), np.asarray(b_rows),
                              sigma_power=sigma_power, theta_ref=theta_ref)


def run_press(
    seed: int,
    *,
    device: str,
    n_grid: int,
    ticks: int,
    substeps: int,
    dt: float,
    v_max: float,
    tau_y: float,
    eta: float,
    density: float,
    bulk: float,
    dough_size: tuple[float, float, float],
    ramp_ticks: int = RAMP_TICKS,
    creep: tuple[float, int, int] | None = None,
) -> tuple[list[dict], float]:
    """One full-travel press; returns (per-tick records, plate_area).

    creep=(v_creep, creep_ticks, accel_ticks) engages at v_creep, holds it for
    creep_ticks, then ramps to v_max over accel_ticks; creep=None keeps the
    plain linear ramp (probe_design_covariance relies on it).
    """
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    pos, vol0, floor = block(grid, size=dough_size, ppc=2, seed=seed)
    solver = Solver(grid=grid, device=device).load_particles(pos, vol0)
    solver.set_material(newtonian(eta=eta, density=density, bulk_modulus=bulk).with_yield(tau_y))
    solver.add_plane((0, 0, floor), (0, 0, 1), "sticky")

    cx = cy = grid.grid_lim * 0.5
    col_w, col_d, col_h = dough_size
    plate_half = (0.5 * col_w + 0.015, 0.5 * col_d + 0.015, 0.6 * grid.dx)
    plate_area = 4.0 * plate_half[0] * plate_half[1]
    dt_ctrl = dt * substeps
    backend = WarpMPMBackend(solver=solver)

    z = floor + col_h + plate_half[2]
    z_floor = floor + plate_half[2] + 0.003
    tool = backend.attach_tool((cx, cy, z), plate_half, velocity=(0, 0, 0))

    records = []
    for tick in range(ticks + 1):
        # engaging at full speed slams the first grid layer and the impulse
        # transient (~ band mass * v / dt_ctrl, grows with resolution) can
        # exceed small stop-force targets outright
        if creep is None:
            v_down_cmd = 0.0 if tick == 0 else v_max * min(1.0, tick / ramp_ticks)
        else:
            v_creep, creep_ticks, accel_ticks = creep
            if tick <= creep_ticks:
                v_down_cmd = v_creep * min(1.0, tick / 4.0)
            else:
                v_down_cmd = v_creep + (v_max - v_creep) * min(1.0, (tick - creep_ticks) / accel_ticks)
        z_new = max(z - v_down_cmd * dt_ctrl, z_floor)
        vz = (z_new - z) / dt_ctrl
        down_speed = max(0.0, -vz)
        if tick > 0:
            backend.set_tool_kinematics(tool, center=(cx, cy, z), velocity=(0, 0, vz))
            backend.reset_tool_force(tool)
            backend.step(dt, substeps)
        z = z_new

        plate_bottom = z - plate_half[2]
        f_grid = max(float(backend.get_tool_reaction(tool, dt_ctrl)[2]), 0.0) if tick > 0 else 0.0
        v = solver.v()
        L = solver.L()
        vol = solver.vol()
        gd, q = equivalent_shear_rate(L)
        records.append({
            "tick": tick,
            "t": tick * dt_ctrl,
            "z": z,
            "down_speed": down_speed,
            "F_grid": f_grid,
            "X1": float(np.sum((q / np.maximum(gd, 1e-12)) * vol)),
            "X2": float(np.sum(q * vol)),
            "P_grav": float(np.sum(density * (-G_MAG) * v[:, 2] * vol)),
            "KE": float(0.5 * density * np.sum(vol * np.sum(v * v, axis=1))),
            "strain": (col_h - (plate_bottom - floor)) / col_h,
            "depth_mm": max(0.0, col_h - (plate_bottom - floor)) * 1e3,
            "gap_mm": (plate_bottom - floor) * 1e3,
        })
    return records, plate_area


def derive_target(
    records: list[dict],
    f_target: float,
    plate_area: float,
    sigma_power: float,
    theta_ref: tuple[float, float],
    seed: int,
) -> dict:
    """Evaluate one stop-force target as a prefix of a recorded press."""
    stop = next((i for i, r in enumerate(records) if i > 0 and r["F_grid"] >= f_target), None)
    prefix = records if stop is None else records[: stop + 1]
    active = [
        r for r in prefix
        if r["tick"] > ENGAGE_TICKS and r["down_speed"] > 1e-5
        and r["F_grid"] > 0.01 * f_target and np.isfinite(r["X2"]) and r["X2"] > 1e-14
    ]
    fit = _fit_power_balance(active, sigma_power=sigma_power, theta_ref=theta_ref)
    peak = max(r["F_grid"] for r in prefix)
    tau_y, eta = theta_ref
    summary = {
        "seed": seed,
        "fit_method": "power_balance",
        "f_target": f_target,
        "target_pressure_pa": f_target / plate_area,
        "target_pressure_kpa": f_target / plate_area / 1000.0,
        "tail_pressure_pa": peak / plate_area,
        "tail_pressure_kpa": peak / plate_area / 1000.0,
        "plate_area_m2": plate_area,
        "peak_grid_force": peak,
        "final_depth_mm": prefix[-1]["depth_mm"],
        "final_strain_pct": 100.0 * prefix[-1]["strain"],
        "ticks_run": len(prefix) - 1,
        "active_rows": len(active),
        "target_reached": stop is not None,
        "hit_floor": bool(prefix[-1]["gap_mm"] <= 3.1),
    }
    if fit["ok"]:
        fixed, fitn = fit["fixed_noise"], fit["fit_noise"]
        summary.update({
            "tau_y_hat": float(fit["theta"][0]),
            "eta_hat": float(fit["theta"][1]),
            "tau_y_err": abs(float(fit["theta"][0]) / tau_y - 1.0),
            "eta_err": abs(float(fit["theta"][1]) / eta - 1.0),
            "law_err": _law_error(fit["theta"], theta_ref, fixed["AtA"]),
            # noise-induced spread of the law error in the probe's own metric:
            # sqrt(E||A dtheta||^2) / ||A theta*|| with dtheta ~ N(0, s^2 (AtA)^-1)
            "law_err_std": float(np.sqrt(2.0) * sigma_power
                                 / max(np.sqrt(np.asarray(theta_ref) @ fixed["AtA"]
                                               @ np.asarray(theta_ref)), 1e-30)),
            **_noise_recovery_rms(fit, theta_ref, sigma_power,
                                  seed=seed * 7919 + int(round(f_target * 1000))),
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
        summary.update({**FIT_FAIL_FIELDS, "reason": fit["reason"]})
    return {"summary": summary, "fit": fit}


def _target_key(value: float) -> float:
    return round(float(value), 8)


def _ordered_summary_keys(summaries: list[dict]) -> list[str]:
    keys, seen = [], set()
    for summary in summaries:
        for key in summary:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _finite_percentiles(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    p25 = np.full(samples.shape[1], np.nan)
    median = np.full(samples.shape[1], np.nan)
    p75 = np.full(samples.shape[1], np.nan)
    for i in range(samples.shape[1]):
        col = samples[:, i]
        col = col[np.isfinite(col)]
        if len(col) > 0:
            p25[i], median[i], p75[i] = np.percentile(col, [25.0, 50.0, 75.0])
    return p25, median, p75


def _write_rows(out_dir: Path, stem: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    (out_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2, default=float))
    with open(out_dir / f"{stem}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    results: list[dict],
    press_by_seed: dict[int, list[dict]],
    out_dir: Path,
    *,
    sigma_power: float,
    tau_y: float,
    eta: float,
    single_targets: list[float],
    dough_size: tuple[float, float, float],
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
    by_seed_force = {
        seed: {_target_key(r["summary"]["f_target"]): r for r in results if r["summary"]["seed"] == seed}
        for seed in seeds
    }

    _write_rows(out_dir, "results", summaries, fieldnames=_ordered_summary_keys(summaries))
    for seed, records in press_by_seed.items():
        np.savez_compressed(
            out_dir / f"seed_{seed}_press_records.npz",
            data=np.array([[rec[k] for k in records[0]] for rec in records], dtype=float),
            keys=np.array(list(records[0].keys())),
        )

    def sgl(seed: int, f_target: float, key: str):
        return by_seed_force[seed][_target_key(f_target)]["summary"][key]

    def band(f_targets: list[float], key: str):
        return _finite_percentiles([[sgl(s, ft, key) for ft in f_targets] for s in seeds])

    def band_plot(filename, title, xlabel, ylabel, curves, true_value=None, log_y=False):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        xmax = 1.0
        for x, (p25, med, p75), color, style, label in curves:
            ok = np.isfinite(x) & np.isfinite(med)
            ax.fill_between(x[ok], p25[ok], p75[ok], color=color, alpha=0.16, linewidth=0)
            ax.plot(x[ok], med[ok], style, lw=2.1, color=color, label=label)
            if ok.any():
                xmax = max(xmax, float(x[ok].max()) * 1.05)
        if log_y:
            ax.set_yscale("log")
        elif true_value is None:
            ax.set_ylim(bottom=0.0)
        if true_value is not None:
            ax.axhline(true_value, color="black", linestyle=":", lw=1.8, label="true")
            ax.margins(y=0.08)
        ax.set_xlim(0.0, xmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    # per-target series, sorted by target pressure
    singles = sorted(single_targets)
    P_real = band(singles, "tail_pressure_kpa")[1]
    rel_tau = band(singles, "rel_std_tau_y_fixed")
    pressure_rows = [{
        "target_pressure_kpa": sgl(seeds[0], ft, "target_pressure_kpa"),
        "realized_pressure_kpa": float(P_real[i]),
        "n_seeds": len(seeds),
        "rel_std_tau_y": rel_tau[1][i],
        "rel_std_tau_y_p25": rel_tau[0][i],
        "rel_std_tau_y_p75": rel_tau[2][i],
    } for i, ft in enumerate(singles)]
    _write_rows(out_dir, "pressure_results", pressure_rows)

    xlabel = "realized plate pressure  (kPa)"
    band_plot("relative_uncertainty_vs_pressure.png", "Yield Stress Uncertainty", xlabel,
              r"yield-stress uncertainty  $\sqrt{C_{\tau_y\tau_y}} / \tau_y$",
              [(P_real, rel_tau, "tab:blue", "o-", "median across seeds")], log_y=True)
    band_plot("viscosity_uncertainty_vs_pressure.png", "Viscosity Uncertainty", xlabel,
              r"viscosity uncertainty  $\sqrt{C_{\eta\eta}} / \eta$",
              [(P_real, band(singles, "rel_std_eta_fixed"), "tab:orange", "o-",
                "median across seeds")], log_y=True)

    # law-space recovery: how wrong the identified flow curve is over the rates the
    # press visited, next to how well the same fit predicts the measured power. The
    # gap between the two curves is the (numerical) dissipation the model absorbed;
    # the blow-up at the smallest pressures is the wrong-action regime. An
    # exact-interpolation fit (rows <= 2) has a machine-zero residual, which is
    # not a measurement — masked off the log plot.
    resb = band(singles, "fit_relres")
    res_med = np.where(resb[1] > 1e-6, resb[1], np.nan)
    lawb = band(singles, "law_err")
    band_plot("law_error_vs_pressure.png",
              "Identified-Law Error over the Probed Rates", xlabel,
              r"relative law error  $\|A(\hat\theta - \theta^*)\|\,/\,\|A\theta^*\|$",
              [(P_real, lawb, "tab:blue", "o-", "identified law vs truth"),
               (P_real, (res_med, res_med, res_med), "0.45", ":", "power-prediction residual")],
              log_y=True)

    def abs_pct(series, true_value, rank_med):
        # |median relative error| in percent (tiny floor only so log axes never
        # see zero). Rank-deficient (single-row) fits are minimum-norm
        # tie-breaks, not estimates — masked.
        med = np.maximum(np.abs(series[1] / true_value - 1.0) * 100.0, 1e-2)
        med = np.where(rank_med >= 2, med, np.nan)
        return (med, med, med)

    rank_med = band(singles, "rank")[1]
    tau_err = abs_pct(band(singles, "tau_y_hat"), tau_y, rank_med)
    eta_err = abs_pct(band(singles, "eta_hat"), eta, rank_med)
    band_plot("parameter_error_vs_pressure.png",
              "Parameter Recovery Error", xlabel, "absolute parameter error  (%)",
              [(P_real, tau_err, "tab:blue", "o-", r"$\tau_y$"),
               (P_real, eta_err, "tab:orange", "s-", r"$\eta$")],
              log_y=True)

    print(f"\nwrote sweep outputs to {out_dir}")
    print(f"realized pressure range: {np.nanmin(P_real):.3f} to {np.nanmax(P_real):.3f} kPa")


def run(args) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    dough_size = tuple(args.dough_size)
    seeds = [args.seed] if args.seed is not None else list(dict.fromkeys(int(s) for s in args.seeds))
    theta_ref = (args.tau_y, args.eta)

    targets, seen = [], set()
    for target in args.targets:
        key = _target_key(target)
        if key not in seen:
            targets.append(float(target))
            seen.add(key)

    results, press_by_seed = [], {}
    for seed in seeds:
        print(f"\n######## seed {seed}: one press, {args.ticks} ticks ########", flush=True)
        records, plate_area = run_press(
            seed, device=args.device, n_grid=args.n_grid, ticks=args.ticks,
            substeps=args.substeps, dt=args.dt, v_max=args.v_max,
            tau_y=args.tau_y, eta=args.eta, density=args.density, bulk=args.bulk,
            dough_size=dough_size, creep=(V_CREEP, CREEP_TICKS, ACCEL_TICKS),
        )
        press_by_seed[seed] = records
        for f_target in targets:
            result = derive_target(records, f_target, plate_area, args.sigma_power, theta_ref, seed)
            s = result["summary"]
            print(
                f"target={s['target_pressure_kpa']:.2f} kPa, reached={s['tail_pressure_kpa']:.2f} kPa, "
                f"rows={s['active_rows']}, theta=({s['tau_y_hat']:.1f}, {s['eta_hat']:.1f}), "
                f"rel_std=({s['rel_std_tau_y_fixed']:.3g}, {s['rel_std_eta_fixed']:.3g})",
                flush=True,
            )
            results.append(result)

    write_outputs(
        results, press_by_seed, OUT,
        sigma_power=args.sigma_power, tau_y=args.tau_y, eta=args.eta,
        single_targets=list(args.targets),
        dough_size=dough_size, seeds=seeds,
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep applied pressure and plot parameter covariance.")
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    parser.add_argument("--targets", nargs="+", type=float,
                        default=[45.0, 33.75, 24.75, 21.375, 18.0, 16.2, 14.625, 13.05,
                                 11.25, 9.45, 7.875, 6.75],
                        help="Target plate forces in N. Default spans 0.3-2 kPa (realized up "
                             "to ~2.5), dense below 1 kPa where identification turns on; the "
                             "gentle-touch engagement keeps even the 0.3-kPa prefix populated.")
    parser.add_argument("--n-grid", type=int, default=128,
                        help="Grid cells per side. 128 keeps the absorbed numerical dissipation "
                             "small (law error ~6-15%%); 48 with --dt 1e-4 --substeps 20 is the "
                             "fast-but-biased iteration mode.")
    parser.add_argument("--ticks", type=int, default=280,
                        help="Control ticks per press. 240 (38 mm travel) reaches the 2 kPa default "
                             "target with margin, well inside the power-balance model's validity.")
    parser.add_argument("--substeps", type=int, default=80)
    parser.add_argument("--dt", type=float, default=2.5e-5)
    parser.add_argument("--v-max", type=float, default=0.08, help="Downward press speed in m/s.")
    parser.add_argument("--sigma-power", type=float, default=0.05,
                        help="Assumed std of power-balance noise b, in Watts, for covariance plots.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Particle initialization seeds for median and seed-band plots.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run one particle initialization seed. Overrides --seeds.")
    parser.add_argument("--tau-y", type=float, default=200.0)
    parser.add_argument("--eta", type=float, default=40.0)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--bulk", type=float, default=9.0e5)
    parser.add_argument("--dough-size", nargs=3, type=float, default=(0.12, 0.12, 0.06),
                        metavar=("W", "D", "H"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
