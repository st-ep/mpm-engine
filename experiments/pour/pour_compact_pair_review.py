"""Review the completed 320-cell pair without launching or fitting anything.

Optional waiting runs under systemd, independently of the paused sweep workers.
This reports a grid-placement check; it cannot establish spatial convergence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.pour.pour_compact_verification import ROOT, OUT, digest, verify_hashes, read_curve, write

PHASES = [0., .5]


def review():
    protocol_path = OUT / "refinement_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    verify_hashes(protocol["input_sha256"])
    results, curves, sources = [], [], [Path(__file__), protocol_path]
    for phase in PHASES:
        directory = OUT / f"refined_n320_phase{phase:g}_sdf384"
        path = directory / "result.json"
        result = json.loads(path.read_text())
        verify_hashes(result["input_sha256"])
        expected = dict(eta_pa_s=protocol["eta_pa_s"], n_grid=320, extent_m=.35,
                        phase_cells=phase, sdf_resolution=384, episode="09-04-60-2s",
                        initial_volume_ml=300., source_wall="sticky", receiver_wall="separable")
        for key, value in expected.items():
            if result[key] != value:
                raise ValueError(f"Unexpected {key} in {path}")
        t, amounts, count = read_curve(directory)
        if count != result["particle_count"]:
            raise ValueError("Particle count disagrees with time history")
        if not np.isclose(amounts[-1, 1], result["receiver_ml"], rtol=0, atol=1e-8):
            raise ValueError("Reported endpoint disagrees with the saved curve")
        final = directory / "09-04-60-2s/final_n320.npz"
        with np.load(final) as state:
            for field in ["x", "v", "vol"]:
                values = state[field]
                if len(values) != count or not np.isfinite(values).all():
                    raise ValueError(f"Invalid final particle state: {field}")
            physical_volume_ml = float(np.sum(state["vol"], dtype=np.float64) * 1e6)
            if abs(physical_volume_ml - 300.) > .01:
                raise ValueError("Particle rest volume differs from the specified fill")
        tail = amounts[t >= t[-1] - .5, 1]
        tail_span = float(np.ptp(tail))
        error = result["receiver_ml"] - 159.
        results.append(dict(phase_cells=phase, receiver_ml=result["receiver_ml"],
                            reference_error_ml=error, within_goal=abs(error) <= 7.,
                            within_hard_limit=abs(error) <= 10.,
                            source_depletion_ml=result["source_depletion_ml"],
                            outside_ml=result["outside_ml"],
                            tail_span_ml=tail_span, settled=tail_span <= .2,
                            boundary_clearance_m=result["minimum_fluid_boundary_clearance_m"],
                            boundary_clear=result["minimum_fluid_boundary_clearance_m"] > 3 * result["dx_m"],
                            final_particle_state_finite=True, particle_count=count,
                            particle_rest_volume_ml=physical_volume_ml,
                            elapsed_s=result["elapsed_s"]))
        curves.append((t, amounts))
        sources.extend([path, directory / "09-04-60-2s/metrics.csv", final])
    if not np.array_equal(curves[0][0], curves[1][0]):
        raise ValueError("Runs used different motion timestamps")
    difference = curves[1][1][:, 1] - curves[0][1][:, 1]
    phase_difference = abs(results[1]["receiver_ml"] - results[0]["receiver_ml"])
    findings = []
    if phase_difference > protocol["numerical_max_difference_ml"]:
        findings.append("Grid placement changes the endpoint by more than the declared 3 mL screen.")
    if not all(r["within_hard_limit"] for r in results):
        findings.append("At least one reference prediction exceeds the 10 mL physical-error limit.")
    if not all(r["settled"] and r["boundary_clear"] and r["outside_ml"] <= 2 for r in results):
        findings.append("At least one settling, boundary or outside-volume check needs attention.")
    if not findings:
        findings.append("The pair passes these limited screens; finer-resolution verification remains necessary.")
    summary = dict(status="Current pair completed; larger simulations remain on hold",
                   reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                   eta_pa_s=protocol["eta_pa_s"], initial_volume_ml=300.,
                   measured_reference_ml=159., reference_used_for="Post-prediction consistency check only",
                   calibration_pour_count=1, six_validation_outcomes_used=False,
                   parameter_fitting_performed=False, simulations_launched=False,
                   spatial_convergence_established=False, independent_validation=False,
                   robot_table_released=False, phase_difference_ml=phase_difference,
                   phase_screen_passed=phase_difference <= protocol["numerical_max_difference_ml"],
                   receiver_curve_rms_difference_ml=float(np.sqrt(np.mean(difference**2))),
                   receiver_curve_max_difference_ml=float(np.abs(difference).max()),
                   runs=results, findings=findings,
                   input_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (t, amounts), r in zip(curves, results):
        ax.plot(t - 1., amounts[:, 1], label=f"Grid placement {r['phase_cells']:g}: {r['receiver_ml']:.1f} mL")
    ax.axhline(159., color="k", ls="--", lw=1, label="Reported reference: 159 mL (check only)")
    ax.set(xlabel="Seconds after tilt command", ylabel="Simulated receiver amount (mL)",
           title="Exact 60° replay: two grid placements at 1.09 mm cell size")
    ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.text(.5, .015, "One-video weak viscosity fixed · no endpoint fitting · no robot table released", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.03,1,1)); fig.savefig(OUT / "pair_review.png", dpi=160); plt.close(fig)
    write(OUT / "pair_review.json", summary)
    text = ["# Review of the current MPM pair", "", "Larger simulations remain on hold. No robot table is released.", "",
            "| Grid placement (cells) | Receiver (mL) | Error against 159 mL |", "|---|---:|---:|"]
    text += [f"| {r['phase_cells']:g} | {r['receiver_ml']:.2f} | {r['reference_error_ml']:+.2f} |" for r in results]
    text += ["", f"Endpoint difference between placements: {phase_difference:.2f} mL.", ""] + findings
    text += ["", "This is a same-resolution grid-placement check. Spatial convergence and independent physical validation remain unestablished.",
             "No parameters were fitted, no validation outcomes were used for fitting, and no further simulations were launched."]
    (OUT / "pair_review.md").write_text("\n".join(text) + "\n")
    return summary


def wait_for_pair():
    hold = json.loads((OUT / "compute_review_hold.json").read_text())
    deadline = time.monotonic() + 4 * 3600
    expected = [OUT / f"refined_n320_phase{p:g}_sdf384/result.json" for p in PHASES]
    while not all(p.exists() for p in expected):
        for index, supervisor in enumerate(hold["paused_supervisors"]):
            if expected[index].exists():
                continue
            for pid in supervisor["children"]:
                path = Path(f"/proc/{pid}/status")
                if not path.exists() or "State:\tZ" in path.read_text():
                    raise RuntimeError(f"Simulation {pid} exited without a result; inspect its log")
        if time.monotonic() > deadline:
            raise TimeoutError("Pair not completed within four hours; no simulations changed")
        time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    try:
        write(OUT / "pair_review_status.json", dict(status="waiting" if args.wait else "reviewing", simulations_launched=False))
        if args.wait:
            wait_for_pair()
        result = review()
        write(OUT / "pair_review_status.json", dict(status="completed", findings=result["findings"], simulations_launched=False))
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        write(OUT / "pair_review_status.json", dict(status="failed", error=str(exc), simulations_launched=False))
        raise
