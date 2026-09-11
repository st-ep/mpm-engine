"""Bounded original-60-degree pair for a verified isolated transfer candidate.

Only two diagnostic replays are allowed. No angle search or physical fit follows
automatically. A failed initial-settling gate aborts before the pour begins.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
SCALES = [1., .5]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path, required=True)
parser.add_argument('--prepare', action='store_true')
parser.add_argument('--case', type=float, choices=SCALES)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
SOURCE = args.source_dir.resolve()
WORK = SOURCE.parent
sys.path.insert(0, str(SOURCE))
from warpmpm.kernels import mpm_utils
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from experiments.pour import pour_navier_reference as reference
from experiments.pour.pour_coulomb_overnight import numerical_problem


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(p)


def result_path(scale):
    return WORK / 'reference_replays/replays/original-separable' / f'recorded60_bNA_n160_phase0_dt{scale:g}_mu0.117000/result.json'


def validate():
    path = WORK / 'reference_protocol.json'
    assert digest(path) == (WORK / 'reference_protocol.sha256').read_text().strip()
    protocol = json.loads(path.read_text())
    for rel, expected in protocol['input_sha256'].items():
        assert digest(ROOT / rel) == expected, rel
    return protocol


def prepare():
    path = WORK / 'reference_protocol.json'
    if path.exists():
        return validate()
    base = WORK.parent / 'compensated_position_20260907/reference_protocol.json'
    protocol = json.loads(base.read_text())
    assert digest(base) == base.with_suffix('.sha256').read_text().strip()
    proto_path = WORK / 'prototype_provenance.json'
    proto = json.loads(proto_path.read_text())
    analytic_path = WORK / 'analytic_review.json'
    analytic = json.loads(analytic_path.read_text())
    assert analytic['bounded_settling_check_supported']
    settle_path = WORK / 'settling_dt1/review.json'
    settle = json.loads(settle_path.read_text())
    assert settle['settling_threshold_passed'] and settle['frozen_inputs_and_raw_outputs_checked']
    assert settle['outside_ml_after_quiet_frame'] <= .1
    assert json.loads((WORK / 'gpu_check.json').read_text())['gpu_pipeline_stable']
    for device in ['cpu', 'cuda0']:
        check = json.loads((WORK / f'benchmark_{device}.json').read_text())
        assert check['all_passed'] and check['prototype_manifest_sha256'] == digest(proto_path)
    frozen = {**protocol['input_sha256'], **proto['prototype_sha256'],
              **analytic['input_sha256'], **settle['evidence_sha256']}
    files = [Path(__file__), base, proto_path, analytic_path, settle_path,
             WORK / 'settling_dt1/protocol.json', WORK / 'gpu_check.json']
    for name in ['pour_transfer_shear_check.py', 'pour_transfer_state_check.py',
                 'pour_transfer_blend_review.py', 'pour_transfer_settling_check.py',
                 'pour_transfer_settling_review.py', 'pour_aflip_blend_prototype.py']:
        files.append(ROOT / 'experiments/pour' / name)
    frozen.update({str(p.relative_to(ROOT)): digest(p) for p in files})
    protocol.update(purpose=__doc__, input_sha256=frozen,
        numerical_change='AFLIP residual-velocity blend exp(log(.99)*dt/reference_dt), PIC advection, stable volume/EOS and compensated positions',
        reference_alpha=.99, beta=0., physical_parameters_refitted=[],
        target_search_authorized=False, production_accuracy_accepted=False,
        coarse_analytical_amplitude_threshold_passed=False,
        implementation_gate='Analytical temporal consistency and refinement, CPU/GPU state and freefall checks, unchanged cup-settling threshold',
        gpu_indices=[0, 1], total_controller_timeout_s=2400, per_run_timeout_s=1800,
        settling_speed_threshold_m_s=reference.twin.SETTLE_SPEED,
        caveat='Coarse analytical shear-amplitude error still exceeds 5%; this pair is diagnostic only.')
    write(path, protocol)
    (WORK / 'reference_protocol.sha256').write_text(digest(path) + '\n')
    (WORK / 'reference_logs').mkdir(exist_ok=False)
    return validate()


def run_case(scale, device):
    protocol = validate()
    actual = Path(mpm_utils.__file__).resolve()
    assert actual.is_relative_to(SOURCE)
    old_project = reference.twin.project_out_of_solid
    gate = {}

    def checked_project(x, v, *positional, **keywords):
        if not gate:
            norms = np.linalg.norm(v, axis=1)
            mean = float(norms.mean())
            gate.update(mean_speed_m_s=mean, max_speed_m_s=float(norms.max()),
                        threshold_m_s=protocol['settling_speed_threshold_m_s'],
                        passed=bool(np.isfinite(norms).all() and mean < protocol['settling_speed_threshold_m_s']))
            write(result_path(scale).parent / 'settling_gate.json', gate)
            if not gate['passed']:
                raise RuntimeError('Initial settling failed; no pour will be run')
        return old_project(x, v, *positional, **keywords)

    reference.OUT = WORK / 'reference_replays'
    reference.twin.project_out_of_solid = checked_project
    try:
        reference.run(SimpleNamespace(source_friction=.117, grid=160,
            wall='original-separable', slip_mm=None, angle=None, max_angle=60.,
            phase=0., dt_scale=scale, device=device))
    finally:
        reference.twin.project_out_of_solid = old_project
    path = result_path(scale)
    result = json.loads(path.read_text())
    assert not numerical_problem(result) and result['particle_count'] == 229280
    assert gate.get('passed', False)
    validate()
    result.update(numerical_correction_protocol_sha256=digest(WORK / 'reference_protocol.json'),
                  actual_kernel_module=str(actual), identification_changed=False,
                  physical_parameters_refitted=[], initial_settling_gate=gate)
    write(path, result)


def report():
    protocol = validate()
    cases = []
    for scale in SCALES:
        path = result_path(scale)
        result = json.loads(path.read_text())
        assert result['numerical_correction_protocol_sha256'] == digest(WORK / 'reference_protocol.json')
        with (path.parent / '09-04-60-2s/metrics.csv').open() as stream:
            records = list(csv.DictReader(stream))
        counts = np.array([[int(r[k]) for k in ['n_src', 'n_rcv', 'n_air_spill']] for r in records])
        assert len(records) == 773 and np.all(counts >= 0) and np.all(counts.sum(1) == 229280)
        assert abs(300 * counts[-1, 1] / 229280 - result['receiver_ml']) < 1e-10
        cases.append(dict(dt_scale=scale, corrected_receiver_ml=result['receiver_ml'],
                          outside_ml=result['outside_ml'], tail_variation_ml=result['tail_variation_ml'],
                          result_sha256=digest(path)))
    difference = cases[1]['corrected_receiver_ml'] - cases[0]['corrected_receiver_ml']
    comparison = dict(cases=cases, half_minus_full_ml=difference,
        timestep_pair_within_1ml=abs(difference) <= protocol['full_pour_timestep_difference_limit_ml'],
        full_temporal_convergence_established=False, goal_accuracy_established=False,
        validation_outcomes_used=[], parameters_refitted=[])
    write(WORK / 'reference_comparison.json', comparison)
    write(WORK / 'reference_status.json', dict(stage='completed_needs_review', **comparison))
    print(json.dumps(comparison, indent=2), flush=True)


def launch():
    protocol = prepare()
    with (WORK / 'reference_controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = dict(zip(SCALES, protocol['gpu_indices']))
        active = {}
        environment = dict(os.environ, MUJOCO_GL='egl', OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        start = time.monotonic()
        try:
            while pending or any(p.poll() is None for p, _, _ in active.values()):
                validate()
                if time.monotonic() - start > protocol['total_controller_timeout_s']:
                    raise RuntimeError('Diagnostic pair reached its time limit')
                for scale, gpu in list(pending.items()):
                    occupied = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
                    if occupied:
                        continue
                    if result_path(scale).parent.exists():
                        raise RuntimeError('Prior output preserved; refusing overwrite')
                    log = (WORK / 'reference_logs' / f'dt{scale:g}.log').open('x')
                    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)),
                        '--source-dir', str(SOURCE), '--case', str(scale), '--device', f'cuda:{gpu}'],
                        cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    active[scale] = (process, log, time.monotonic())
                    del pending[scale]
                exits = {str(s): p.poll() for s, (p, _, _) in active.items()}
                if any(code not in [None, 0] for code in exits.values()):
                    raise RuntimeError(f'Replay failed: {exits}')
                for process, _, began in active.values():
                    if process.poll() is None and time.monotonic() - began > protocol['per_run_timeout_s']:
                        raise RuntimeError('Per-run time limit')
                write(WORK / 'reference_status.json', dict(stage='running',
                    pending_gpu_by_scale=pending, exits=exits,
                    pids={str(s): p.pid for s, (p, _, _) in active.items()},
                    elapsed_s=time.monotonic() - start))
                time.sleep(10)
            assert all(p.returncode == 0 for p, _, _ in active.values())
            report()
        except Exception as error:
            write(WORK / 'reference_status.json', dict(stage='stopped_needs_review', reason=str(error)))
            raise
        finally:
            for process, log, _ in active.values():
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                log.close()


if __name__ == '__main__':
    if args.prepare:
        prepare()
    elif args.case is not None:
        run_case(args.case, args.device)
    else:
        launch()
