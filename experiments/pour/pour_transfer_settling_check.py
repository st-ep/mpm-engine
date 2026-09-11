"""Bounded rest-state check, preserving the original pour's scene and material.

This runs only the unchanged initial settling plus one quiet replay frame. It
is not a pour prediction or a command table. No receiver endpoint is consumed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path, required=True)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--dt-scale', type=float, choices=[1., .5], default=1.)
args = parser.parse_args()
source = args.source_dir.resolve()
sys.path.insert(0, str(source))
import numpy as np
from warpmpm.kernels import mpm_utils
from experiments.pour import pour_navier_reference as reference


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert Path(mpm_utils.__file__).resolve().is_relative_to(source)
    manifest = source.parent / 'prototype_provenance.json'
    proto = json.loads(manifest.read_text())
    for rel, sha in proto['prototype_sha256'].items():
        assert digest(ROOT / rel) == sha, rel
    work = source.parent / f'settling_dt{args.dt_scale:g}'
    work.mkdir(exist_ok=False)
    twin = reference.twin
    fixed = [Path(__file__), Path(reference.__file__), Path(twin.__file__),
             manifest, reference.GEOMETRY, reference.OBSERVATIONS,
             *[reference.REFERENCE / name for name in ['states.jsonl', 'actions.jsonl', 'meta.json']]]
    hashes = {**proto['prototype_sha256'], **{str(p.relative_to(ROOT)): digest(p) for p in fixed}}
    protocol = dict(purpose=__doc__, input_sha256=hashes,
                    actual_kernel_module=str(Path(mpm_utils.__file__).resolve()),
                    eta_pa_s=reference.ETA, source_coulomb_friction=.117,
                    grid=160, volume_ml=300., dt_scale=args.dt_scale,
                    maximum_settling_s=twin.SETTLE_MAX_S,
                    mean_speed_threshold_m_s=twin.SETTLE_SPEED, replay_frame_cap=1,
                    physical_parameters_refitted=[], validation_outcomes_used=[],
                    exploratory_only=True, production_accuracy_accepted=False)
    write(work / 'protocol.json', protocol)
    start = time.monotonic()
    trace = dict(stage='settling', samples=[], ticks=0)
    original_run = twin.run
    original_project = twin.project_out_of_solid

    def speeds(v):
        norms = np.linalg.norm(v, axis=1)
        return dict(mean_speed_m_s=float(norms.mean()),
                    p95_speed_m_s=float(np.percentile(norms, 95)),
                    max_speed_m_s=float(norms.max()), finite=bool(np.isfinite(norms).all()))

    def monitored_run(*positional, **keywords):
        previous_solver = twin.Solver

        class MonitoringSolver(previous_solver):
            def step(self, *step_args, **step_kwargs):
                result = super().step(*step_args, **step_kwargs)
                if trace['stage'] == 'settling':
                    trace['ticks'] += 1
                    if trace['ticks'] % 12 == 0:
                        sample = dict(time_s=trace['ticks'] / twin.FPS, **speeds(self.v()))
                        trace['samples'].append(sample)
                        write(work / 'progress.json', trace)
                        print('settling check:', json.dumps(sample), flush=True)
                return result

        twin.Solver = MonitoringSolver
        try:
            return original_run(*positional, **dict(keywords, frames=1))
        finally:
            twin.Solver = previous_solver

    def monitored_project(x, v, *positional, **keywords):
        initial = trace['stage'] == 'settling'
        if initial:
            trace['before_projection'] = speeds(v)
        result = original_project(x, v, *positional, **keywords)
        if initial:
            trace['after_projection'] = speeds(result[1])
            trace['projected_particles'] = int(result[2])
            trace['stage'] = 'one_quiet_replay_frame'
            write(work / 'progress.json', trace)
        return result

    twin.run, twin.project_out_of_solid = monitored_run, monitored_project
    reference.OUT = work / 'forward'
    try:
        reference.run(SimpleNamespace(source_friction=.117, grid=160,
            wall='original-separable', slip_mm=None, angle=None, max_angle=60.,
            phase=0., dt_scale=args.dt_scale, device=args.device))
    finally:
        twin.run, twin.project_out_of_solid = original_run, original_project
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha, rel
    results = list((work / 'forward').rglob('result.json'))
    assert len(results) == 1
    forward = json.loads(results[0].read_text())
    passed = (trace['before_projection']['finite'] and
              trace['before_projection']['mean_speed_m_s'] < twin.SETTLE_SPEED)
    result = dict(protocol_sha256=digest(work / 'protocol.json'),
                  **trace, stage='completed', settling_threshold_passed=passed,
                  particle_count=forward['particle_count'],
                  outside_ml_after_quiet_frame=forward['outside_ml'],
                  simulation_result_sha256=digest(results[0]),
                  elapsed_s=time.monotonic() - start,
                  full_pour_performed=False, goal_accuracy_established=False)
    write(work / 'review.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
