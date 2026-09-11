"""Controlled angle-response audit with the numerical grid fixed to the reference.

This diagnostic overrides only the scene-to-grid translation inside its own process.
Existing planning results and calibration inputs are retained unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from examples import pour_recorded_twin as twin

from experiments.pour.pour_angle_plan import (
    ETA,
    GEOMETRY,
    MESH,
    OUTPUT,
    REFERENCE,
    ROOT,
    angle_name,
    fingerprint,
)
from experiments.pour.pour_angle_sweep import build_motion, write_planned_episode

AUDIT = ROOT / "out/pour_angle_curve_audit"


def offset_for(episode, scene, n_grid):
    ep = twin.load_episode(episode, twin.PRE_ROLL, twin.HOLD_SECONDS)
    arm = twin.RecordedPanda(ep, MESH, height=64, width=64, max_geom=4000,
                             cup_reference_pos=scene["cup_reference_pos"],
                             cup_reference_quat=scene["cup_reference_quat"])
    receiver = np.array([*scene["receiver_xy"], scene["table_z"]])
    offset = twin.world_to_mpm_offset(arm, receiver, twin.GRID_LIM / n_grid)
    arm.close()
    return offset


def run(angle, n_grid=192, device="cuda:1", mode="reference"):
    g = json.loads(GEOMETRY.read_text())
    scene = {key: g[key] for key in
             ["cup_reference_pos", "cup_reference_quat", "receiver_xy", "table_z"]}
    episode = OUTPUT / "episodes" / angle_name(angle)
    if not episode.exists():
        motion, ep = build_motion(REFERENCE, MESH)
        episode = AUDIT / "episodes" / angle_name(angle)
        write_planned_episode(REFERENCE, episode, motion, ep, angle)
    original = offset_for(episode, scene, n_grid)
    reference = offset_for(REFERENCE, scene, n_grid)
    root = AUDIT / f"{mode}_n{n_grid}"
    result_path = root / episode.name / "result.json"
    provenance = dict(original_planner=fingerprint(angle), n_grid=n_grid, grid_mode=mode,
                      reference_offset=reference.tolist(), original_offset=original.tolist(),
                      audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["provenance"] != provenance:
            raise ValueError(f"Stale audit result at {result_path}")
        return result
    twin.OUT_ROOT = root
    original_offset_function = twin.world_to_mpm_offset
    if mode == "reference":
        twin.world_to_mpm_offset = lambda arm, receiver, dx: reference.copy()
    try:
        simulation = twin.run(episode, device=device, n_grid=n_grid, video=False,
                              side_by_side=False, rebake=True, eta=ETA, volume_ml=300, **scene)
    finally:
        twin.world_to_mpm_offset = original_offset_function
    first, last = simulation["rows"][0], simulation["rows"][-1]
    count = sum(first[k] for k in ["n_src", "n_rcv", "n_air_spill"])
    result = dict(angle_deg=angle, eta_pa_s=ETA, receiver_ml=300 * last["n_rcv"] / count,
                  source_depletion_ml=300 * (1 - last["n_src"] / count),
                  outside_ml=300 * last["n_air_spill"] / count, provenance=provenance)
    result_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", type=float, nargs="+", required=True)
    parser.add_argument("--n-grid", type=int, default=192)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--mode", choices=["reference", "per-angle"], default="reference")
    args = parser.parse_args()
    for angle in args.angles:
        print(json.dumps(run(angle, args.n_grid, args.device, args.mode)), flush=True)
