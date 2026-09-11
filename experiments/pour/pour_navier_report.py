"""Plot existing one-pour calibration results without changing parameters."""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.pour.pour_navier_reference import ETA, OBSERVATIONS, OUT


def main():
    results = [json.loads(p.read_text()) for p in
               sorted((OUT / "replays/wet-traction").glob("recorded60_*/result.json"))]
    if not results:
        raise ValueError("No completed recorded-pour simulations")
    protocol = json.loads((OUT / "calibration_protocol.json").read_text())
    observed = dict(np.load(OBSERVATIONS))
    valid = np.isfinite(observed["rcv_vol"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].plot(observed["t"][valid], 1e6 * observed["rcv_vol"][valid],
                 color="black", alpha=.7, label="Identification video: optical estimate")
    summary = []
    for result in results:
        if result["eta_pa_s"] != ETA or result["validation_outcomes_used"]:
            raise ValueError("Unexpected calibration provenance")
        folder = OUT / "replays/wet-traction" / result["name"]
        with (folder / "09-04-60-2s/metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        # Twin includes its fixed 1 s preroll before the recorded tilt command.
        from examples.pour_recorded_twin import PRE_ROLL
        t = np.array([float(r["t"]) for r in rows]) - PRE_ROLL
        v = 300 * np.array([float(r["n_rcv"]) for r in rows]) / result["particle_count"]
        label = f"MPM b={result['slip_length_mm']:g} mm, grid {result['n_grid']}"
        line, = axes[0].plot(t, v, label=label)
        axes[1].scatter(result["slip_length_mm"], result["receiver_ml"],
                        color=line.get_color(), label=label,
                        marker="o" if result["n_grid"] == 160 else "s")
        fit_window = result["weak_fit_window_s"]
        mask = valid & (observed["t"] >= fit_window[0]) & (observed["t"] <= fit_window[1])
        error = np.interp(observed["t"][mask], t, v) - 1e6*observed["rcv_vol"][mask]
        summary.append(dict(name=result["name"], slip_length_mm=result["slip_length_mm"],
                            n_grid=result["n_grid"], receiver_ml=result["receiver_ml"],
                            endpoint_error_ml=result["receiver_ml"]-protocol["calibration_endpoint_ml"],
                            optical_window_rms_ml=float(np.sqrt(np.mean(error**2))),
                            outside_ml=result["outside_ml"], tail_variation_ml=result["tail_variation_ml"]))
    for ax in axes:
        ax.axhline(protocol["calibration_endpoint_ml"], color="gray", linestyle="--",
                   label="Same-pour measured endpoint: 159 mL")
        ax.set_ylabel("Receiver volume (mL)")
        ax.grid(alpha=.2)
    axes[0].set_xlabel("Time from recorded tilt command (s)")
    axes[0].set_title("Exact 60° recording replay")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].set_xlabel("Effective source-wall slip (mm)")
    axes[1].set_title("Calibration trials; viscosity fixed")
    axes[1].set_xlim(.1, 4.2)
    fig.suptitle(f"One-pour calibration diagnostics · η = {ETA:.2f} Pa·s\n"
                 "159 mL is a calibration target; transfer accuracy is unvalidated", fontsize=11)
    fig.savefig(OUT / "calibration_diagnostics.png", dpi=170)
    plt.close(fig)
    (OUT / "calibration_diagnostics.json").write_text(json.dumps(dict(
        eta_pa_s=ETA, identification_episode="09-04-60-2s", samples=summary,
        validation_outcomes_used=[], optical_window_is_diagnostic_only=True,
        parameter_selection_uses="159 mL endpoint of the same identification pour only"), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
