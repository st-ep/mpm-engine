"""After the measured-rim probe, run the original wall at matched eta and grid."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_navier_calibration"


def main():
    geometry_result = OUT / "measured_lip_probe/comparison.json"
    deadline = time.monotonic() + 1800
    while not geometry_result.exists():
        state = subprocess.run(["systemctl", "--user", "is-active", "pour-measured-lip-probe.service"],
                               capture_output=True, text=True).stdout.strip()
        if state != "active" or time.monotonic() > deadline:
            raise RuntimeError("Geometry probe did not finish successfully; review before further runs")
        time.sleep(10)
    print("Geometry probe complete. Running original separable/Coulomb contact at eta=3.4392377844275503 and grid=160.", flush=True)
    log = OUT / "logs/original_separable_n160.log"
    with log.open("w") as stream:
        subprocess.run([sys.executable, "-u", "experiments/pour/pour_navier_reference.py",
                        "--grid", "160", "--wall", "original-separable", "--device", "cuda:1"],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
    original_path = OUT / "replays/original-separable/recorded60_bNA_n160_phase0_dt1/result.json"
    baseline_path = OUT / "replays/wet-traction/recorded60_b1.000000_n160_phase0_dt1/result.json"
    original, baseline = (json.loads(p.read_text()) for p in [original_path, baseline_path])
    if original["eta_pa_s"] != baseline["eta_pa_s"] or original["particle_count"] != baseline["particle_count"]:
        raise ValueError("Unmatched comparison")
    comparison = dict(eta_pa_s=original["eta_pa_s"], n_grid=160,
                      original_contact_receiver_ml=original["receiver_ml"],
                      b1_traction_receiver_ml=baseline["receiver_ml"],
                      measured_rim_b1=json.loads(geometry_result.read_text()),
                      source_coulomb_coefficient=.05, fitted_parameters=[], validation_outcomes_used=[],
                      status="Matched-setting diagnostics only; no parameter selection or commands")
    (OUT / "original_wall_comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
