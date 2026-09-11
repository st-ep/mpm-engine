"""Check whether source/receiver volume normalization cancels readout bias.

This is a diagnostic, not an accepted replacement for the receiver observation.
No final-volume measurement enters either normalization. A common multiplicative
bias cancels in the two-cup ratio; unequal or pose-dependent biases do not.
The formulas omit liquid in flight, so during pouring they are approximations.
Settled endpoints assume no spill or unaccounted retained liquid.

Run from the repository root:
  python experiments/pour/pour_conservation_audit.py --episode 09-04-60-2s
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]


def transferred_from_ratio(source, receiver, initial_ml):
    """Normalize two nonnegative volume proxies with a common unknown gain.

    Inputs must use the same units; output is mL. Missing/invalid proxies remain
    missing rather than being turned into zero or filled from conservation.
    """
    source, receiver = np.broadcast_arrays(np.asarray(source, float),
                                          np.asarray(receiver, float))
    total = source + receiver
    valid = (np.isfinite(source) & np.isfinite(receiver) & (source >= 0)
             & (receiver >= 0) & (total > 0))
    result = np.full(source.shape, np.nan)
    np.divide(initial_ml * receiver, total, out=result, where=valid)
    return result


def transferred_from_source(source, initial_source_proxy, initial_ml):
    """Source depletion scaled by a known initial fill, without clipping.

    Negative values or subsequent decreases expose measurement inconsistency.
    During a pour this estimates receiver plus in-flight volume, not receiver
    volume alone. The source's optical gain must stay constant with pose.
    """
    if not np.isfinite(initial_source_proxy) or initial_source_proxy <= 0:
        raise ValueError("Initial source proxy must be positive and finite")
    return initial_ml * (1 - np.asarray(source, float) / initial_source_proxy)


def finite_median(values):
    values = np.asarray(values)
    return float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else None


def run(episode, initial_ml=300.0, baseline=(-2.0, -0.25), output=None):
    if not np.isfinite(initial_ml) or initial_ml <= 0:
        raise ValueError("Initial volume must be positive and finite")
    folder = REPO / "out" / "pour_wf" / episode
    observations_path = folder / "observations.npz"
    with np.load(observations_path) as data:
        obs = {key: data[key] for key in data.files}
    t = obs["t"]
    receiver = obs["rcv_vol"] * 1e6
    baseline_mask = (t >= baseline[0]) & (t <= baseline[1])
    # The observation payload stores return acknowledgement on the pour clock.
    return_done = float(obs["t_ret_done"])
    settled = (t >= return_done + 0.15) & (t <= return_done + 0.65)
    report = {
        "episode": episode,
        "initial_volume_ml": initial_ml,
        "observations_sha256": hashlib.sha256(observations_path.read_bytes()).hexdigest(),
        "baseline_window_s": list(baseline),
        "settled_window_s": [return_done + 0.15, return_done + 0.65],
        "final_measurements_used_in_normalization": False,
        "accepted_for_identification": False,
        "assumptions": ["same gain for two-cup ratio", "pose-independent source gain for depletion",
                        "in-flight liquid omitted from these diagnostic curves"],
        "channels": {},
    }
    curves = {"t": t, "receiver_raw_ml": receiver}
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, receiver, color="black", label="Receiver readout")
    for field, label, colour in [("src_vol", "Deep source", "tab:blue"),
                                ("src_vol_loose", "Loose source", "tab:orange")]:
        source = obs[field] * 1e6
        before = finite_median(source[baseline_mask])
        if before is None or before <= 0:
            raise ValueError(f"No valid positive baseline for {field}")
        ratio = transferred_from_ratio(source, receiver, initial_ml)
        depletion = transferred_from_source(source, before, initial_ml)
        curves[f"{field}_ratio_ml"] = ratio
        curves[f"{field}_depletion_ml"] = depletion
        report["channels"][field] = {
            "baseline_source_proxy_ml": before,
            "settled_source_proxy_ml": finite_median(source[settled]),
            "settled_receiver_proxy_ml": finite_median(receiver[settled]),
            "settled_ratio_prediction_ml": finite_median(ratio[settled]),
            "settled_source_depletion_prediction_ml": finite_median(depletion[settled]),
            "minimum_source_depletion_ml": float(np.nanmin(depletion)),
        }
        axes[0].plot(t, ratio, color=colour, label=f"{label}: two-cup ratio")
        axes[0].plot(t, depletion, "--", color=colour, alpha=0.7,
                     label=f"{label}: initial-fill normalization")
        axes[1].plot(t, source + receiver, color=colour, label=f"{label} + receiver")
        axes[1].axhline(before, color=colour, ls=":", alpha=0.6)

    manifest_path = folder / "recording_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        endpoint = manifest.get("final_receiver_volume_ml")
        report["comparison_only_reported_endpoint_ml"] = endpoint
        if endpoint is not None:
            axes[0].axhline(endpoint, color="tab:green", ls="--",
                            label="Reported endpoint (comparison only)")
    axes[0].set(ylabel="Transferred volume [mL]",
                title="Conservation audit: diagnostic curves, no endpoint fitting")
    axes[1].axhline(initial_ml, color="black", ls="--", label="Known initial volume")
    axes[1].set(xlabel="Time after pour command [s]", ylabel="Sum of raw proxies [mL]",
                title="A constant common gain would preserve the sum after the stream lands")
    for axis in axes:
        axis.axvspan(*baseline, color="grey", alpha=0.08)
        axis.axvline(return_done, color="grey", lw=1, ls=":")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output = Path(output) if output else folder / "conservation_audit"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "conservation.png", dpi=170)
    plt.close(fig)
    np.savez_compressed(output / "diagnostic_curves.npz", **curves)
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default="09-04-60-2s")
    parser.add_argument("--initial-ml", type=float, default=300.0)
    parser.add_argument("--baseline", type=float, nargs=2, default=(-2.0, -0.25))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.episode, args.initial_ml, tuple(args.baseline), args.output)
