"""Does the brink weak closure recover the known viscosity of saved MPM runs?

Uses the saved source particle inventory directly, removing optical measurement
and source-to-receiver transport estimation from this diagnostic. It performs no
new MPM run and does not replace the viscosity identified from the real video.
"""
from pathlib import Path
import json
import numpy as np

from experiments.pour.pour_compact_verification import ROOT, OUT, digest, read_curve, verify_hashes, write
from experiments.pour.pour_weakform_identify import brink_forcing, fit_eta_integrated


def main():
    destination = OUT / "closure_recovery"
    destination.mkdir(exist_ok=False)
    baseline = ROOT / "out/pour_weakform_recovery/identified"
    identity = json.loads((baseline / "identify.json").read_text())
    # Only pose/time arrays enter; no real liquid observation is passed to the fit.
    with np.load(baseline / "observations.npz") as source:
        poses = {k: source[k] for k in ["t", "cup_pos", "cup_quat"]}
    window = identity["t_fit"]
    protocol = dict(purpose="Synthetic recovery diagnostic using saved MPM source inventory",
                    real_liquid_observations_used=False, experimental_endpoints_used=False,
                    six_validation_outcomes_used=False, new_simulations_run=False,
                    production_viscosity_changed=False, t_fit=window,
                    inventory="Source depletion from particle counts; not receiver volume",
                    flight_correction="Unnecessary: source inventory is directly available",
                    limitation="Diagnoses combined closure/discretization mismatch; does not establish continuum MPM accuracy")
    paths = [Path(__file__), baseline / "identify.json", baseline / "observations.npz",
             ROOT / "experiments/pour/pour_weakform_identify.py",
             ROOT / "experiments/pour/pour_perception.py"]
    cases = []
    for phase in [0., .5]:
        directory = OUT / f"refined_n320_phase{phase:g}_sdf384"
        path = directory / "result.json"
        result = json.loads(path.read_text())
        verify_hashes(result["input_sha256"])
        t, volume, _ = read_curve(directory)
        t = t - 1.  # Existing replay clock includes one second of pre-roll.
        synthetic = dict(poses)
        synthetic["rcv_vol"] = np.interp(poses["t"], t, 300. - volume[:, 0]) * 1e-6
        # The existing assembly calls this field rcv_vol, but with no flight
        # correction it simply supplies V0 minus the known source inventory.
        forcing = brink_forcing(synthetic, 300., tuple(window), eta_flight=None)
        eta, _, keep, dv, cf = fit_eta_integrated(forcing)
        row = dict(phase_cells=phase, mpm_input_eta_pa_s=result["eta_pa_s"],
                   weak_closure_recovered_eta_pa_s=eta,
                   recovered_to_input_ratio=eta / result["eta_pa_s"],
                   fit_rms_ml=float(np.sqrt(np.mean((dv[keep] - cf[keep] / eta)**2))),
                   actual_window_source_depletion_ml=float(dv[-1]),
                   closure_window_prediction_at_mpm_eta_ml=float(cf[-1] / result["eta_pa_s"]),
                   n_kept=int(keep.sum()), n_samples=len(keep),
                   fit_sample_times_s=forcing["t"][[0, -1]].tolist(),
                   maximum_hydrostatic_film_head_mm=float(np.max(forcing["h_tip"]) * 1000))
        cases.append(row)
        np.savez_compressed(destination / f"phase{phase:g}.npz", **forcing,
                            fit_keep=keep, fit_delta_source_ml=dv, fit_cumulative_forcing=cf)
        paths += [path, directory / "09-04-60-2s/metrics.csv"]
    protocol["input_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    report = dict(protocol=protocol, cases=cases,
                  interpretation="A discrepancy means the hydrostatic brink closure and this discretized MPM do not assign the same viscosity to the same simulated flow. No coefficient correction is applied.")
    write(destination / "results.json", report)
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
