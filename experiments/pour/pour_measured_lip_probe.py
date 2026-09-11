"""One causal MPM test of artificial outlet widening, with no parameter fit."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from experiments.pour import pour_navier_reference as reference
from experiments.pour.pour_measured_lip_geometry import measured_lip_field, measured_lip_solid


def main():
    twin = reference.twin
    output = reference.ROOT / "out/pour_navier_calibration/measured_lip_probe"
    output.mkdir(parents=True, exist_ok=False)
    geometry = json.loads(reference.GEOMETRY.read_text())
    receiver = np.r_[geometry["receiver_xy"], geometry["table_z"]]
    offset = reference.fixed_scene(geometry)[0]
    dx = .35 / 160
    extra_wall, extra_base = twin.collision_extras(dx)
    protocol = dict(
        purpose="Causal source-geometry comparison with completed b=1 mm baseline; not endpoint calibration",
        eta_pa_s=reference.ETA, n_grid=160, slip_length_mm=1., initial_volume_ml=300.,
        measured_endpoints_used=[], validation_outcomes_used=[],
        source_rim_width_measured_mm=1000*twin.SPEC.rim_width,
        source_rim_width_before_padding_mm=1000*(twin.SPEC.rim_width+extra_wall),
        padding_transition_start_m=twin.SPEC.spout_z0,
        padding_transition_end_m=twin.SPEC.rim_z-twin.SPEC.rim_blend,
        transition_rule="Existing spout depth to existing rim-blend boundary, smoothstep; no new dimension fitted",
        receiver_geometry="unchanged", cavity_geometry="unchanged",
        b_interpretation="Retained solely for controlled comparison; not accepted as physical slip or a final model",
        expected_direction="Unknown; result must be observed, not assumed",
        input_sha256={str(p.relative_to(reference.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [reference.ROOT / "experiments/pour/pour_measured_lip_probe.py",
                                reference.ROOT / "experiments/pour/pour_measured_lip_geometry.py"]})
    (output / "probe_protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    with measured_lip_field():
        source_sdf = twin.build_cup_sdf(twin.SPEC, res=256, margin=.010,
                                        extra_wall=extra_wall, extra_base=extra_base)
    old_solver, old_audit, old_project, old_out = twin.Solver, twin.cup_audit, twin.project_out_of_solid, reference.OUT

    class SourceGeometrySolver(old_solver):
        def add_sdf_collider(self, *positional, **keywords):
            index = getattr(self, "_geometry_source_count", 0)
            self._geometry_source_count = index + 1
            if index == 0:
                positional = (source_sdf, *positional[1:])
            return super().add_sdf_collider(*positional, **keywords)

    def audit(x, pos, quat, h, extras, w2m):
        values = old_audit(x, pos, quat, h, extras, w2m)
        if np.allclose(pos, receiver, atol=1e-8, rtol=0):
            return values
        local = twin.world_to_local(x, np.asarray(pos) + w2m, quat)
        n_solid = int((measured_lip_solid(local, twin.SPEC, *extras) < 0.).sum())
        return (n_solid, *values[1:])

    def project(x, v, pos, quat, spec, **kwargs):
        if np.allclose(pos, receiver + offset, atol=1e-8, rtol=0):
            return old_project(x, v, pos, quat, spec, **kwargs)
        with measured_lip_field():
            return old_project(x, v, pos, quat, spec, **kwargs)

    twin.Solver, twin.cup_audit, twin.project_out_of_solid, reference.OUT = SourceGeometrySolver, audit, project, output
    try:
        reference.run(argparse.Namespace(grid=160, phase=0., slip_mm=1., angle=None,
                                         dt_scale=1., device="cuda:1", wall="wet-traction"))
    finally:
        twin.Solver, twin.cup_audit, twin.project_out_of_solid, reference.OUT = old_solver, old_audit, old_project, old_out
    path = next(output.glob("replays/wet-traction/*/result.json"))
    result = json.loads(path.read_text())
    result["input_sha256"].update(protocol["input_sha256"])
    result["source_geometry_variant"] = "measured_lip_probe"
    result["source_geometry_protocol"] = str(output / "probe_protocol.json")
    result["status"] = "Source geometry diagnostic only; b retained from baseline for causal comparison"
    path.write_text(json.dumps(result, indent=2)+"\n")
    baseline = json.loads((old_out / "replays/wet-traction/recorded60_b1.000000_n160_phase0_dt1/result.json").read_text())
    comparison = dict(probe=protocol, baseline_receiver_ml=baseline["receiver_ml"],
                      corrected_rim_receiver_ml=result["receiver_ml"],
                      change_ml=result["receiver_ml"]-baseline["receiver_ml"],
                      outside_ml=result["outside_ml"], tail_variation_ml=result["tail_variation_ml"],
                      status="Geometry diagnostic only; no boundary calibration or angle release")
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
