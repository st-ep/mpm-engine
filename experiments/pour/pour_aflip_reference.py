"""Exploratory original60 timestep pair using published unblended AFLIP.

The transfer update was selected from analytical shear and no-force checks.
Weak viscosity, contact, geometry and robot motion remain fixed. The coarse
analytical amplitude error remains above5%; this is not production acceptance.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'out/pour_physics_audit/aflip_20260907'
sys.path.insert(0, str(WORK / 'isolated_src'))
from warpmpm.kernels import mpm_utils

import argparse
import csv
import fcntl
import json
import os
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from experiments.pour import pour_navier_reference as reference
from experiments.pour.pour_coulomb_overnight import digest, write, numerical_problem, case_path

SCALES = [1., .5]


def result_path(scale):
    return WORK / 'reference_replays/replays/original-separable' / f'recorded60_bNA_n160_phase0_dt{scale:g}_mu0.117000/result.json'


def validate():
    p = WORK / 'reference_protocol.json'
    assert digest(p) == (WORK / 'reference_protocol.sha256').read_text().strip()
    protocol = json.loads(p.read_text())
    for relative, expected in protocol['input_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Frozen input changed: {relative}')
    return protocol


def prepare():
    p = WORK / 'reference_protocol.json'
    if p.exists():
        return validate()
    base = WORK.parent / 'compensated_position_20260907'
    base_protocol = json.loads((base / 'reference_protocol.json').read_text())
    assert digest(base / 'reference_protocol.json') == (base / 'reference_protocol.sha256').read_text().strip()
    proto = json.loads((WORK / 'prototype_provenance.json').read_text())
    for device in ['cpu', 'cuda0']:
        check = json.loads((WORK / f'benchmark_{device}.json').read_text())
        assert check['all_passed']
        assert check['prototype_manifest_sha256'] == digest(WORK / 'prototype_provenance.json')
    frozen = {**base_protocol['input_sha256'], **proto['prototype_sha256']}
    for path in [Path(__file__), base / 'reference_protocol.json',
                 WORK / 'prototype_provenance.json', WORK / 'benchmark_cpu.json',
                 WORK / 'benchmark_cuda0.json',
                 ROOT / 'experiments/pour/pour_compensated_position_benchmark.py']:
        frozen[str(path.relative_to(ROOT))] = digest(path)
    analytic = json.loads((WORK / 'analytic_review.json').read_text())
    assert analytic['exploratory_recorded_pour_test_supported']
    assert json.loads((WORK / 'gpu_check.json').read_text())['gpu_pipeline_stable']
    frozen.update(analytic['input_sha256'])
    for extra in [WORK / 'analytic_review.json', WORK / 'gpu_check.json',
                  ROOT / 'experiments/pour/pour_aflip_prototype.py',
                  ROOT / 'experiments/pour/pour_aflip_checks.py',
                  ROOT / 'experiments/pour/pour_aflip_analytic_review.py',
                  ROOT / 'experiments/pour/pour_shear_decay_benchmark.py',
                  ROOT / 'experiments/pour/pour_stable_volume_gpu_check.py']:
        frozen[str(extra.relative_to(ROOT))] = digest(extra)
    protocol = dict(base_protocol)
    protocol.update(purpose=__doc__, input_sha256=frozen,
                    numerical_change='AFLIP alpha1 beta0; base log-volume/EOS and compensated positions retained',
                    aflip_alpha=1., aflip_beta=0., production_accuracy_accepted=False,
                    coarse_analytical_amplitude_threshold_passed=False,
                    implementation_gate='Analytical shear temporal consistency and refinement, inviscid preservation, CPU/GPU state and freefall checks; coarse shear amplitude still fails5percent',
                    gpu_indices=[0,1], total_controller_timeout_s=2400,
                    per_run_timeout_s=1800, target_search_authorized=False)
    write(p, protocol)
    (WORK / 'reference_protocol.sha256').write_text(digest(p)+'\n')
    (WORK / 'reference_logs').mkdir(exist_ok=True)
    return validate()


def run_case(scale, device):
    validate()
    actual = str(Path(mpm_utils.__file__).resolve())
    assert actual.startswith(str(WORK / 'isolated_src'))
    reference.OUT = WORK / 'reference_replays'
    reference.run(SimpleNamespace(source_friction=.117, grid=160,
        wall='original-separable', slip_mm=None, angle=None, max_angle=60.,
        phase=0., dt_scale=scale, device=device))
    p = result_path(scale)
    r = json.loads(p.read_text())
    if numerical_problem(r) or r['particle_count'] != 229280:
        raise RuntimeError('Capture or particle checks failed')
    validate()
    r.update(numerical_correction_protocol_sha256=digest(WORK / 'reference_protocol.json'),
             actual_kernel_module=actual, identification_changed=False,
             physical_parameters_refitted=[])
    write(p, r)


def report():
    validate()
    rows = []
    for scale in SCALES:
        p = result_path(scale)
        r = json.loads(p.read_text())
        assert r['numerical_correction_protocol_sha256'] == digest(WORK / 'reference_protocol.json')
        with (p.parent / '09-04-60-2s/metrics.csv').open() as stream:
            metrics = list(csv.DictReader(stream))
        counts = np.array([[int(x[k]) for k in ['n_src','n_rcv','n_air_spill']] for x in metrics])
        assert np.all(counts >= 0) and np.all(counts.sum(1) == 229280)
        assert abs(300*counts[-1,1]/229280-r['receiver_ml']) < 1e-12
        old = json.loads((case_path(60.,dt=scale) / 'result.json').read_text())
        rows.append(dict(dt_scale=scale, original_receiver_ml=old['receiver_ml'],
                         corrected_receiver_ml=r['receiver_ml'], outside_ml=r['outside_ml'],
                         tail_variation_ml=r['tail_variation_ml'], result_sha256=digest(p)))
    delta = rows[1]['corrected_receiver_ml'] - rows[0]['corrected_receiver_ml']
    comparison = dict(cases=rows, half_minus_full_ml=delta,
                      timestep_pair_within_1ml=abs(delta)<=1.,
                      full_temporal_convergence_established=False,
                      goal_accuracy_established=False, validation_outcomes_used=[],
                      parameters_refitted=[])
    write(WORK / 'reference_comparison.json', comparison)
    write(WORK / 'reference_status.json', dict(stage='completed_needs_review', **comparison))
    print(json.dumps(comparison, indent=2), flush=True)


def launch():
    protocol = prepare()
    with (WORK / 'reference_controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = dict(zip(SCALES, protocol['gpu_indices']))
        active = {}
        env = dict(os.environ, MUJOCO_GL='egl', PYTHONPATH=str(ROOT),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        start = time.monotonic()
        try:
            while pending or any(proc.poll() is None for proc,_,_ in active.values()):
                validate()
                if time.monotonic()-start > protocol['total_controller_timeout_s']:
                    raise RuntimeError('Bounded pair controller time limit')
                for scale, gpu in list(pending.items()):
                    occupied = subprocess.check_output(['nvidia-smi','-i',str(gpu),
                        '--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).strip()
                    if occupied:
                        continue
                    if result_path(scale).parent.exists():
                        raise RuntimeError('Prior output preserved; refuse overwrite')
                    log = (WORK / 'reference_logs' / f'dt{scale:g}.log').open('w')
                    proc = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                        '--case',str(scale),'--device',f'cuda:{gpu}'], cwd=ROOT,
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[scale] = (proc,log,time.monotonic())
                    del pending[scale]
                exits = {str(s): proc.poll() for s,(proc,_,_) in active.items()}
                if any(code not in [None,0] for code in exits.values()):
                    raise RuntimeError(f'Replay failed: {exits}')
                for proc,_,began in active.values():
                    if proc.poll() is None and time.monotonic()-began > protocol['per_run_timeout_s']:
                        raise RuntimeError('Per-run time limit')
                write(WORK / 'reference_status.json', dict(stage='running',
                    pending_gpu_by_scale=pending, exits=exits,
                    pids={str(s):p.pid for s,(p,_,_) in active.items()},
                    elapsed_s=time.monotonic()-start))
                time.sleep(10)
            # Check final exits even if both children completed during the sleep.
            assert all(p.returncode == 0 for p,_,_ in active.values())
            report()
        except Exception as error:
            write(WORK / 'reference_status.json', dict(stage='stopped_needs_review', reason=str(error)))
            raise
        finally:
            for proc,log,_ in active.values():
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--case', type=float, choices=SCALES)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.case is not None:
        run_case(args.case, args.device)
    else:
        launch()
