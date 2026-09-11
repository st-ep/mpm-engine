"""Finish existing calibration checks, then run at most two timing probes.

No calibration replay is launched here. Wait for the two already running
checks; on any failure stop, preserving results. No bulk search or handoff.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'out/pour_physics_audit/aflip_blend_time_20260907'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact', required=True, type=float)
    parser.add_argument('--grid-service', default='pour-aflip-time-recalibration0272-n192-20260908.service')
    parser.add_argument('--time-service', default='pour-aflip-time-recalibration0272-halfdt-20260908.service')
    args = parser.parse_args()
    candidate = WORK / f'candidate_mu{args.contact:.6f}'
    candidate.mkdir(exist_ok=True)
    sequence = candidate / 'two_probe_sequence'
    sequence.mkdir(exist_ok=False)
    scripts = [Path(__file__), *[ROOT/'experiments/pour'/name for name in
        ['pour_transfer_candidate_check.py', 'pour_transfer_timing_probes.py',
         'pour_transfer_probe_review.py', 'pour_transfer_calibration_review.py']]]
    frozen = {str(p.relative_to(ROOT)):digest(p) for p in scripts}
    started = time.monotonic()
    active = []
    protocol = dict(purpose=__doc__, source_contact=args.contact,
        first_probe_angles_deg=[48.37,53.04], maximum_probes=2,
        wall_limit_s=2700, per_probe_limit_s=1800, input_sha256=frozen,
        calibration_jobs_launched=0, bulk_search_queued=False,
        real_error_forecast_saved=False)
    (sequence/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')

    def status(stage, **extra):
        value = dict(stage=stage, updated_unix=time.time(), elapsed_s=time.monotonic()-started,
            bulk_search_queued=False, **extra)
        (sequence/'status.json').write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(value),flush=True)

    def validate():
        if time.monotonic()-started>protocol['wall_limit_s']:
            raise RuntimeError('Bounded sequence reached its wall limit')
        for rel, sha in frozen.items():
            if digest(ROOT/rel)!=sha:
                raise RuntimeError(f'Frozen reviewer/probe changed: {rel}')

    def command(module, *extra):
        return [sys.executable,'-u','-m',f'experiments.pour.{module}',
            '--contact',str(args.contact),*extra]

    def checked_review(stage):
        validate()
        output = candidate/f'{stage}_review.json'
        if output.exists():
            raise FileExistsError('Unexpected pre-existing review; preserve and inspect')
        subprocess.run(command('pour_transfer_candidate_check','--stage',stage),
            cwd=ROOT,check=True,timeout=120)
        review = json.loads(output.read_text())
        passed = review['limited_numerical_checks_passed'] if stage=='all' else (
            review['endpoint_calibration_passed'] and review['comparison_passed'])
        if not passed:
            raise RuntimeError(f'{stage} check failed; no angle probes will be run')
        status(f'{stage}_check_passed')

    env = dict(os.environ, MUJOCO_GL='egl', PYTHONPATH=str(ROOT),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    pending = {'grid':WORK/f'recalibration_mu{args.contact:.6f}_n192_dt1/completion.json',
               'time':WORK/f'recalibration_mu{args.contact:.6f}_n160_dt0.5/completion.json'}
    services = {'grid':args.grid_service,'time':args.time_service}
    try:
        status('waiting_for_existing_checks')
        while pending:
            validate()
            for stage, path in list(pending.items()):
                if path.exists():
                    checked_review(stage)
                    del pending[stage]
                else:
                    state = subprocess.check_output(['systemctl','--user','show',
                        services[stage],'-p','ActiveState','--value'],text=True).strip()
                    if state not in ('active','activating'):
                        raise RuntimeError(f'{services[stage]} ended without a completed result')
            if pending:
                time.sleep(10)
        checked_review('all')
        # Do not share a GPU with another process, including an exiting replay.
        status('waiting_for_free_gpus')
        while True:
            validate()
            processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).strip()
            if not processes:
                break
            time.sleep(10)
        for gpu,angle in enumerate([48.37,53.04]):
            validate()
            log = sequence/f'angle_{angle:.2f}.log'
            stream = log.open('x')
            proc = subprocess.Popen(command('pour_transfer_timing_probes','--angle',str(angle),
                '--device',f'cuda:{gpu}'),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            active.append(dict(proc=proc,stream=stream,angle=angle,started=time.monotonic()))
        status('running_two_probes',pids={str(p['angle']):p['proc'].pid for p in active})
        while active:
            validate()
            for task in list(active):
                rc = task['proc'].poll()
                if rc is None:
                    if time.monotonic()-task['started']>protocol['per_probe_limit_s']:
                        raise RuntimeError('Probe exceeded its time limit')
                    continue
                task['stream'].close()
                if rc:
                    raise RuntimeError(f"Probe at {task['angle']} failed; inspect preserved log")
                subprocess.run(command('pour_transfer_probe_review','--angle',str(task['angle'])),
                    cwd=ROOT,check=True,timeout=120)
                active.remove(task)
                status('probe_reviewed',angle_deg=task['angle'])
            if active:
                time.sleep(10)
        status('ready_for_chat_forecast',probes_completed=2,
               next_action='Review plots and report expected real errors in chat before any bulk run')
    except Exception as exc:
        status('stopped_for_review',error=str(exc))
        raise
    finally:
        for task in active:
            if task['proc'].poll() is None:
                task['proc'].terminate()
                try:
                    task['proc'].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    task['proc'].kill();task['proc'].wait()
            task['stream'].close()


if __name__=='__main__':
    main()
