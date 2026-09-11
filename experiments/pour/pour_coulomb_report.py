"""Report completed original-contact replays; never select parameters."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.pour.pour_navier_reference import ETA, OBSERVATIONS, OUT
from examples.pour_recorded_twin import PRE_ROLL


def plot_completed_prediction():
    """Show simulated search points only; never infer a new command from the plot."""
    prediction_path = OUT / "prediction_60ml_coulomb.json"
    if not prediction_path.exists():
        return
    prediction = json.loads(prediction_path.read_text())
    samples = json.loads((OUT / "coulomb_60ml_samples.json").read_text())
    if prediction["eta_pa_s"] != ETA or prediction["validation_outcomes_used"]:
        raise ValueError("Unexpected prediction provenance")
    for sample in samples:
        if (sample["eta_pa_s"] != ETA or sample["validation_outcomes_used"]
                or sample["source_coulomb_friction"] != prediction["source_coulomb_friction"]
                or sample["n_grid"] != prediction["n_grid"]):
            raise ValueError("Search points do not share the frozen model")
    angles = [s["planned_angle_deg"] for s in samples]
    volumes = [s["receiver_ml"] for s in samples]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.axhspan(59., 61., color="gray", alpha=.12, label="Simulation acceptance: 60 ± 1 mL")
    ax.axhline(60., color="gray", ls="--", linewidth=1)
    ax.scatter(angles, volumes, s=45, label="Completed MPM replays")
    for angle, volume in zip(angles, volumes, strict=True):
        if angle == prediction["command_angle_deg"]:
            continue
        close_to_selected = abs(angle-prediction["command_angle_deg"]) < .5
        ax.annotate(f"{angle:.2f}°: {volume:.2f} mL", (angle, volume),
                    xytext=(12, -26) if close_to_selected else (0, 10),
                    textcoords="offset points", ha="left" if close_to_selected else "center", fontsize=9)
    angle, volume = prediction["command_angle_deg"], prediction["predicted_ml"]
    ax.scatter([angle], [volume], marker="*", s=190, color="tab:green", zorder=4,
               label=f"Selected replay: {angle:.2f}°, {volume:.2f} mL")
    ax.set(xlabel="Commanded angle (degrees)", ylabel="Settled receiver volume (mL)",
           xlim=(min(angles)-.7, max(angles)+.7), ylim=(min(volumes)-7, max(volumes)+10))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"60 mL command search · fixed η = {ETA:.2f} Pa·s, contact μ = {prediction['source_coulomb_friction']:g}\n"
                 "300 mL initial fill · 2 s hold · grid 160 · robot validation pending", fontsize=11)
    fig.savefig(OUT / "coulomb_60ml_search.png", dpi=180)
    plt.close(fig)


def main():
    observations = dict(np.load(OBSERVATIONS))
    paths = sorted((OUT / "replays/original-separable").glob("recorded60*/result.json"))
    results = [json.loads(p.read_text()) for p in paths]
    if not results:
        raise ValueError("No completed original-contact replays")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    rows_out = []
    for path, result in zip(paths, results, strict=True):
        if result["eta_pa_s"] != ETA or result["validation_outcomes_used"]:
            raise ValueError("Unexpected provenance")
        with (path.parent / "09-04-60-2s/metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        t = np.array([float(r["t"]) for r in rows])-PRE_ROLL
        v = 300*np.array([float(r["n_rcv"]) for r in rows])/result["particle_count"]
        mu, grid = result["source_coulomb_friction"], result["n_grid"]
        label = f"MPM μ={mu:g}, grid {grid}"
        line, = axes[0].plot(t, v, label=label)
        axes[1].scatter(mu, result["receiver_ml"], color=line.get_color(),
                        marker="o" if grid == 160 else "s", label=label)
        w = result["weak_fit_window_s"]
        mask = np.isfinite(observations["rcv_vol"]) & (observations["t"] >= w[0]) & (observations["t"] <= w[1])
        error = np.interp(observations["t"][mask], t, v)-1e6*observations["rcv_vol"][mask]
        rows_out.append(dict(name=result["name"], contact_mu=mu, n_grid=grid,
                             receiver_ml=result["receiver_ml"], error_to_calibration_endpoint_ml=result["receiver_ml"]-159.,
                             optical_window_rms_ml=float(np.sqrt(np.mean(error**2))),
                             outside_ml=result["outside_ml"], tail_variation_ml=result["tail_variation_ml"]))
    axes[0].scatter(observations["t"][mask], 1e6*observations["rcv_vol"][mask],
                    color="black", s=9, label="Identification video: weak-fit window")
    for ax in axes:
        ax.axhline(159., color="gray", ls="--", label="Same-pour measurement: 159 mL")
        ax.set_ylabel("Receiver volume (mL)")
        ax.grid(alpha=.2)
    axes[0].set_xlabel("Time from recorded tilt command (s)")
    axes[1].set_xlabel("Effective engine contact coefficient μ")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"60° reference · fixed weak-form η = {ETA:.2f} Pa·s\n"
                 "One-pour simulator calibration diagnostics; transfer remains unvalidated", fontsize=11)
    fig.savefig(OUT / "coulomb_diagnostics.png", dpi=170)
    plt.close(fig)
    (OUT / "coulomb_diagnostics.json").write_text(json.dumps(dict(samples=rows_out,
        validation_outcomes_used=[], optical_comparison_is_diagnostic_only=True), indent=2)+"\n")
    plot_completed_prediction()
    print(json.dumps(rows_out, indent=2))


if __name__ == "__main__":
    main()
