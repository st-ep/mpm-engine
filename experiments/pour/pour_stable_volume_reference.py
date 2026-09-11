"""Two recorded60 MPM checks of the analytically verified volume update.

No parameter fitting and no validation outcomes. The old forward model, source
data and handoff remain untouched. The alternate kernel lives in an isolated
source copy with a separately frozen manifest.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'out/pour_physics_audit/stable_volume_20260907'
sys.path.insert(0,str(WORK/'isolated_src'))

import argparse
import csv
import json
import os
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from experiments.pour import pour_navier_reference as reference
from experiments.pour.pour_coulomb_overnight import digest,write,numerical_problem,case_path
from warpmpm.kernels import mpm_utils

SCALES=[1.,.5]


def path_for(scale):
    return WORK/'reference_replays/replays/original-separable'/f'recorded60_bNA_n160_phase0_dt{scale:g}_mu0.117000/result.json'


def validate():
    p=WORK/'reference_protocol.json'
    if digest(p)!=(WORK/'reference_protocol.sha256').read_text().strip():
        raise RuntimeError('Reference protocol changed')
    protocol=json.loads(p.read_text())
    for relative,expected in protocol['input_sha256'].items():
        if digest(ROOT/relative)!=expected:raise RuntimeError(f'Frozen input changed: {relative}')
    return protocol


def prepare():
    if (WORK/'reference_protocol.json').exists():return validate()
    for name in ['analytic_review.json','gpu_check.json']:
        if not (WORK/name).exists():raise RuntimeError(f'Missing gate: {name}')
    assert json.loads((WORK/'analytic_review.json').read_text())['all_analytic_checks_passed']
    assert json.loads((WORK/'gpu_check.json').read_text())['gpu_pipeline_stable']
    proto=json.loads((WORK/'prototype_provenance.json').read_text())
    frozen={**proto['source_sha256'],**proto['prototype_sha256']}
    paths=[Path(__file__),WORK/'prototype_provenance.json',WORK/'analytic_review.json',WORK/'gpu_check.json',
        reference.GEOMETRY,reference.OBSERVATIONS,reference.MESH,
        Path(reference.__file__),Path(reference.twin.__file__),
        ROOT/'experiments/pour/pour_weakform_recovery.py',ROOT/'experiments/pour/pour_weakform_identify.py',
        ROOT/'experiments/pour/pour_coulomb_overnight.py',
        *[reference.REFERENCE/n for n in ['actions.jsonl','states.jsonl','meta.json']],
        *[case_path(60.,dt=s)/'result.json' for s in SCALES],
        ROOT/'out/pour_navier_calibration/philip_pilot_20260907_200/philip_pilot_handoff.zip']
    frozen.update({str(p.relative_to(ROOT)):digest(p) for p in paths})
    protocol=dict(purpose=__doc__,recording='09-04-60-2s',dt_scales=SCALES,
        eta_pa_s=reference.ETA,source_contact=.117,initial_volume_ml=300.,grid=160,
        extent_m=.35,phase=0.,particle_count=229280,
        geometry_changed=False,identification_changed=False,physical_parameters_refitted=[],
        numerical_change='Separate log-volume state, stable Euler determinant increment and same EOS',
        implementation_gate='Analytical compression max error<0.2%, stable-timestep GPU checks',
        full_pour_timestep_difference_limit_ml=1.,outside_limit_ml=3.,tail_limit_ml=.5,
        same_initialization_rule='Fresh settling at each timestep; same fill, pose and stopping rule',
        validation_outcomes_used=[],measured_endpoints_used=[],target_search_authorized=False,
        per_run_timeout_s=2400,input_sha256=frozen)
    write(WORK/'reference_protocol.json',protocol)
    (WORK/'reference_protocol.sha256').write_text(digest(WORK/'reference_protocol.json')+'\n')
    (WORK/'reference_logs').mkdir(exist_ok=True)
    return validate()


def run_case(scale,device):
    validate()
    actual=str(Path(mpm_utils.__file__).resolve())
    if not actual.startswith(str(WORK/'isolated_src')):raise RuntimeError('Wrong kernel imported')
    reference.OUT=WORK/'reference_replays'
    reference.run(SimpleNamespace(source_friction=.117,grid=160,wall='original-separable',
        slip_mm=None,angle=None,max_angle=60.,phase=0.,dt_scale=scale,device=device))
    path=path_for(scale);r=json.loads(path.read_text())
    if numerical_problem(r) or r['particle_count']!=229280:raise RuntimeError('Capture/particle checks failed')
    validate()
    r.update(stable_volume_protocol_sha256=digest(WORK/'reference_protocol.json'),
        actual_kernel_module=actual,identification_changed=False,physical_parameters_refitted=[])
    write(path,r)


def report():
    validate();rows=[]
    for scale in SCALES:
        p=path_for(scale);r=json.loads(p.read_text());old=json.loads((case_path(60.,dt=scale)/'result.json').read_text())
        assert r['stable_volume_protocol_sha256']==digest(WORK/'reference_protocol.json')
        with (p.parent/'09-04-60-2s/metrics.csv').open() as f:metrics=list(csv.DictReader(f))
        counts=np.array([[int(x[k]) for k in ['n_src','n_rcv','n_air_spill']] for x in metrics])
        assert np.all(counts>=0) and np.all(counts.sum(1)==229280)
        assert abs(300*counts[-1,1]/229280-r['receiver_ml'])<1e-12
        rows.append(dict(dt_scale=scale,old_receiver_ml=old['receiver_ml'],
            stable_volume_receiver_ml=r['receiver_ml'],outside_ml=r['outside_ml'],
            tail_variation_ml=r['tail_variation_ml'],result_sha256=digest(p)))
    delta=rows[1]['stable_volume_receiver_ml']-rows[0]['stable_volume_receiver_ml']
    result=dict(cases=rows,stable_half_minus_full_ml=delta,
        previous_half_minus_full_ml=rows[1]['old_receiver_ml']-rows[0]['old_receiver_ml'],
        timestep_pair_within_1ml=abs(delta)<=1.,validation_outcomes_used=[],
        parameters_refitted=[],goal_accuracy_established=False)
    write(WORK/'reference_comparison.json',result)
    write(WORK/'reference_status.json',dict(stage='completed_needs_review',**result))
    print(json.dumps(result,indent=2),flush=True)


def launch():
    p=prepare()
    import fcntl
    with (WORK/'reference_controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if occupied:raise RuntimeError('GPU process active; no overlapping run launched')
        active={}
        env=dict(os.environ,MUJOCO_GL='egl',PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        try:
            for scale,gpu in zip(SCALES,[0,1]):
                if path_for(scale).parent.exists():raise RuntimeError('Prior output preserved; do not overwrite')
                log=(WORK/'reference_logs'/f'dt{scale:g}.log').open('w')
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--case',str(scale),'--device',f'cuda:{gpu}'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[scale]=(proc,log)
            start=time.monotonic()
            while True:
                validate();exits={str(s):proc.poll() for s,(proc,log) in active.items()}
                write(WORK/'reference_status.json',dict(stage='running',exits=exits,
                    pids={str(s):proc.pid for s,(proc,_) in active.items()},elapsed_s=time.monotonic()-start))
                if any(v not in [None,0] for v in exits.values()):raise RuntimeError(f'Failed replay: {exits}')
                if all(v==0 for v in exits.values()):break
                if time.monotonic()-start>p['per_run_timeout_s']:raise RuntimeError('Reference pair time limit')
                time.sleep(10)
            report()
        except Exception as error:
            write(WORK/'reference_status.json',dict(stage='stopped_needs_review',reason=str(error)))
            raise
        finally:
            for proc,log in active.values():
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                log.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--case',type=float,choices=SCALES);ap.add_argument('--device',default='cuda:0')
    a=ap.parse_args()
    if a.prepare:prepare()
    elif a.case is not None:run_case(a.case,a.device)
    else:launch()
