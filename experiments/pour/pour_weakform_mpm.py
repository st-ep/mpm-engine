"""Forward MPM predictions at the frozen spout-edge weak-form viscosity.

The fitted viscosity is read from its identification artifact. No measured angle
endpoints or empirical angle-control module is imported. Grid translation is
fixed to the 60-degree reference for every planned motion at a given resolution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from examples import pour_recorded_twin as twin

from experiments.pour.pour_angle_sweep import build_motion, write_planned_episode

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "pouring_real_data/09-04-60-2s"
IDENTIFIED = ROOT / "out/pour_weakform_recovery/identified"
OUTPUT = ROOT / "out/pour_weakform_recovery/mpm"
MESH = ROOT / "out/pour_wf/09-04-60-2s/cup_render.obj"


def simulate(angle, grid, device, shift_fraction=0.0):
    identification_path = IDENTIFIED / "identify.json"
    identification = json.loads(identification_path.read_text())
    if (identification["channel"] != "brink-integrated"
            or identification["viscosity_fitted_by_simulator"]
            or identification["measured_endpoints_used"]
            or identification["validation_measurements_used"]):
        raise ValueError("Requires an endpoint-free weak-form identification artifact")
    eta = float(identification["eta"])
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("Invalid weak-form viscosity")
    geometry_path = IDENTIFIED / "geometry.json"
    scene = json.loads(geometry_path.read_text())
    name = f"angle_{angle:06.2f}"
    root = OUTPUT / f"n{grid}_shift{shift_fraction:g}"
    result_path = root / name / "result.json"
    paths = [identification_path, geometry_path, Path(__file__), Path(twin.__file__),
             Path(__file__).with_name("pour_angle_sweep.py"),
             ROOT / "src/warpmpm/geometry/measuring_cup.py",
             REFERENCE / "states.jsonl", REFERENCE / "actions.jsonl", REFERENCE / "meta.json"]
    provenance = dict(angle_deg=angle, eta_pa_s=eta, n_grid=grid,
                      grid_shift_fraction_xz=shift_fraction, initial_volume_ml=300,
                      hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths})
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["provenance"] != provenance:
            raise RuntimeError(f"Stale simulation at {result_path}")
        return result
    motion, reference = build_motion(REFERENCE, MESH)
    episode = OUTPUT / "episodes" / name
    planning = write_planned_episode(REFERENCE, episode, motion, reference, angle)
    ep = twin.load_episode(REFERENCE, twin.PRE_ROLL, twin.HOLD_SECONDS)
    arm = twin.RecordedPanda(ep, MESH, height=64, width=64, max_geom=4000,
                             cup_reference_pos=scene["cup_reference_pos"],
                             cup_reference_quat=scene["cup_reference_quat"])
    original_offset = twin.world_to_mpm_offset
    offset = original_offset(arm, np.array([*scene["receiver_xy"], scene["table_z"]]),
                             twin.GRID_LIM / grid)
    arm.close()
    offset += shift_fraction * twin.GRID_LIM / grid * np.array([1., 0., 1.])
    twin.OUT_ROOT = root
    twin.world_to_mpm_offset = lambda arm, receiver, dx: offset.copy()
    start = time.monotonic()
    try:
        simulation = twin.run(episode, device=device, n_grid=grid, video=False,
                               side_by_side=False, rebake=True, eta=eta, volume_ml=300, **scene)
    finally:
        twin.world_to_mpm_offset = original_offset
    rows = simulation["rows"]
    count = sum(rows[0][k] for k in ["n_src", "n_rcv", "n_air_spill"])
    drift = max(abs(sum(r[k] for k in ["n_src", "n_rcv", "n_air_spill"]) - count)
                for r in rows)
    if drift:
        raise RuntimeError("Particle ledger changed")
    last = rows[-1]
    tail = [r["n_rcv"] for r in rows if r["t"] >= last["t"] - 0.5]
    result = dict(provenance=provenance, planning=planning,
                  receiver_ml=300 * last["n_rcv"] / count,
                  source_depletion_ml=300 * (1 - last["n_src"] / count),
                  outside_ml=300 * last["n_air_spill"] / count,
                  tail_variation_ml=300 * float(np.ptp(tail)) / count,
                  particle_count=count, particle_count_drift=drift,
                  fixed_world_to_grid_offset=offset.tolist(),
                  elapsed_s=time.monotonic()-start,
                  method="forward MPM at frozen time-weak spout-edge viscosity",
                  status="prospective numerical prediction; physical accuracy not established")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angle", type=float, required=True)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shift-fraction", type=float, default=0.)
    args = parser.parse_args()
    print(json.dumps(simulate(args.angle, args.grid, args.device, args.shift_fraction)), flush=True)
