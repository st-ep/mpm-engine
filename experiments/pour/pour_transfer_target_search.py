"""Bounded MPM-only command search; preserve the failed timestep screen.

Critical phase verifies/refines 60 and 100 mL, then stops for review. Full phase
requires those checks, obtains higher-angle anchors, and completes all eight
targets. No fluid validation outcomes or real-error forecasts are read or saved.
"""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

from experiments.pour.pour_transfer_target_plan import linear_proposal
from experiments.pour.pour_coulomb_timestep_followup import gpu_snapshot, eligible

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'out/pour_physics_audit/aflip_blend_time_20260907/candidate_mu0.272000'
WORK = BASE/'table_search'
MODEL = BASE/'target_model_protocol.json'
EXTERNAL = {
    47.32:'pour-aflip-target60-angle47p32-20260908.service',
    52.23:'pour-aflip-target100-angle52p23-20260908.service',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def case_path(angle):
    return BASE/f'target_cases/angle_{angle:.2f}'


def command(module, *args):
    return [sys.executable,'-u','-m','experiments.pour.'+module,'--contact','0.272',*args]


class Search:
    def __init__(self, phase):
        self.phase = phase
        self.model = json.loads(MODEL.read_text())
        self.validate_model()
        self.frozen = {str(p.relative_to(ROOT)):digest(p) for p in [
            MODEL,Path(__file__),ROOT/'experiments/pour/pour_transfer_target_plan.py',
            ROOT/'experiments/pour/pour_transfer_target_case.py',
            ROOT/'experiments/pour/pour_transfer_target_review.py',
            ROOT/'experiments/pour/pour_transfer_diagnostic_review.py',
            ROOT/'experiments/pour/pour_coulomb_timestep_followup.py',
            ROOT/'experiments/pour/pour_coulomb_overnight.py']}
        protocol_path = WORK/f'{phase}_control_protocol.json'
        if protocol_path.exists():
            raise FileExistsError('Controller phase already exists; inspect its actual process/results before resuming')
        write(protocol_path,dict(purpose=__doc__,phase=phase,input_sha256=self.frozen,
            model_deadline_unix=self.model['created_unix']+self.model['wall_budget_s'],
            maximum_new_target_runs=self.model['maximum_new_target_runs'],
            fluid_validation_outcomes_used=[],real_error_forecast_saved=False,
            original_time_check_passed=False,numerical_acceptance_passed=False))
        state_path = WORK/'state.json'
        self.state = json.loads(state_path.read_text()) if state_path.exists() else dict(points={},targets={})
        for angle in [48.37,53.04]:
            self.state['points'][f'{angle:.2f}'] = self.read_case(angle,diagnostic=True)
        for entry in self.state['points'].values():
            assert self.read_case(entry['angle_deg'],entry['kind']=='diagnostic')==entry
        self.active = {}
        self.env = dict(os.environ,PYTHONPATH=str(ROOT),MUJOCO_GL='egl',
                        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

    def validate_model(self):
        assert digest(MODEL)==MODEL.with_suffix('.sha256').read_text().strip()
        assert self.model['eta_pa_s']==3.4392377844275503 and self.model['source_contact']==.272
        assert self.model['forecast_reported_in_chat_before_target_runs']
        assert not self.model['original_time_check_passed'] and not self.model['numerical_acceptance_passed']
        assert not self.model['validation_outcomes_used']
        for rel,expected in self.model['input_sha256'].items():
            assert digest(ROOT/rel)==expected,rel

    def validate(self):
        self.validate_model()
        if time.time()>self.model['created_unix']+self.model['wall_budget_s']:
            raise RuntimeError('Original target-verification wall budget exhausted')
        for rel,expected in self.frozen.items():
            assert digest(ROOT/rel)==expected,rel

    def status(self, stage, **extra):
        write(WORK/'state.json',self.state)
        row=dict(stage=stage,phase=self.phase,updated_unix=time.time(),
            completed_targets=sorted(map(int,self.state['targets'])),
            allocated_target_cases=len(list((BASE/'target_cases').glob('angle_*'))),
            maximum_new_target_runs=self.model['maximum_new_target_runs'],
            numerical_acceptance_passed=False,original_time_check_passed=False,
            active={str(a):{'pid':t['process'].pid,'gpu':t['gpu']} for a,t in self.active.items()},
            **extra)
        write(WORK/'status.json',row)
        print(json.dumps(row),flush=True)

    def read_case(self, angle, diagnostic=False):
        case=BASE/f'diagnostic_timing_probe_{angle:.2f}' if diagnostic else case_path(angle)
        completion=json.loads((case/'completion.json').read_text())
        review_path=case/'independent_review.json'
        if not review_path.exists():
            subprocess.run(command('pour_transfer_diagnostic_review' if diagnostic else 'pour_transfer_target_review',
                '--angle',str(angle)),cwd=ROOT,check=True,timeout=120,stdout=subprocess.DEVNULL)
        review=json.loads(review_path.read_text())
        for rel,expected in review['input_sha256'].items():
            assert digest(ROOT/rel)==expected,rel
        protocol=json.loads((case/'protocol.json').read_text())
        for rel,expected in protocol['input_sha256'].items():
            assert digest(ROOT/rel)==expected,rel
        result_path=ROOT/completion['result_path']
        assert digest(result_path)==completion['result_sha256']
        result=json.loads(result_path.read_text())
        assert review['accounting_finite_state_and_frozen_inputs_passed']
        assert review['angle_deg']==angle and result['planned_angle_deg']==angle
        assert result['eta_pa_s']==self.model['eta_pa_s'] and result['source_coulomb_friction']==.272
        assert result['receiver_ml']==review['receiver_ml']==completion['receiver_ml']
        assert not review['fluid_validation_outcomes_used']
        return dict(angle_deg=angle,receiver_ml=result['receiver_ml'],kind='diagnostic' if diagnostic else 'target',
            result_path=str(result_path.relative_to(ROOT)),result_sha256=digest(result_path),
            review_path=str(review_path.relative_to(ROOT)),review_sha256=digest(review_path))

    def adopt_external(self):
        pending=dict(EXTERNAL)
        self.status('waiting_for_existing_target_checks')
        while pending:
            self.validate()
            for angle,unit in list(pending.items()):
                if (case_path(angle)/'completion.json').exists():
                    self.state['points'][f'{angle:.2f}']=self.read_case(angle)
                    del pending[angle]
                    self.status('existing_target_check_reviewed',angle_deg=angle)
                else:
                    state=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True).strip()
                    if state not in ('active','activating'):
                        raise RuntimeError(f'{unit} ended without a completed result')
            if pending:time.sleep(10)

    def batch(self, angles):
        pending=[]
        for angle in sorted(set(round(a,2) for a in angles)):
            self.validate()
            key=f'{angle:.2f}'
            if key in self.state['points']:continue
            if (case_path(angle)/'completion.json').exists():
                self.state['points'][key]=self.read_case(angle)
            elif case_path(angle).exists():
                raise RuntimeError(f'Existing incomplete case {angle}; preserve and inspect, do not duplicate')
            else:pending.append(angle)
        try:
            while pending or self.active:
                self.validate()
                for angle,task in list(self.active.items()):
                    rc=task['process'].poll()
                    if rc is None:
                        if time.monotonic()-task['started']>1800:
                            raise RuntimeError(f'Case {angle} exceeded its 30-minute limit')
                        continue
                    task['stream'].close()
                    if rc:raise RuntimeError(f'Case {angle} failed; inspect {task["log"]}')
                    self.state['points'][f'{angle:.2f}']=self.read_case(angle)
                    del self.active[angle]
                    self.status('forward_case_reviewed',angle_deg=angle,
                        receiver_ml=self.state['points'][f'{angle:.2f}']['receiver_ml'])
                if pending:
                    occupied={t['gpu'] for t in self.active.values()}
                    for gpu in gpu_snapshot():
                        if not pending:break
                        if gpu['index'] in occupied or not eligible(gpu,dict(gpu_max_utilization_percent=5,gpu_max_used_memory_mib=2048)):
                            continue
                        if len(list((BASE/'target_cases').glob('angle_*')))>=self.model['maximum_new_target_runs']:
                            raise RuntimeError('Original target-verification run budget exhausted')
                        angle=pending.pop(0)
                        log=WORK/f'angle_{angle:.2f}.log';stream=log.open('x')
                        proc=subprocess.Popen(command('pour_transfer_target_case','--angle',str(angle),
                            '--device',f'cuda:{gpu["index"]}'),cwd=ROOT,env=self.env,stdout=stream,stderr=subprocess.STDOUT)
                        self.active[angle]=dict(process=proc,stream=stream,gpu=gpu['index'],
                            started=time.monotonic(),log=str(log))
                        self.status('forward_case_started',angle_deg=angle,pending_angles=pending)
                if pending or self.active:time.sleep(10)
        finally:
            for task in self.active.values():
                if task['process'].poll() is None:
                    task['process'].terminate()
                    try:task['process'].wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        task['process'].kill();task['process'].wait()
                task['stream'].close()
            self.active={}

    def solve(self, targets):
        while any(str(t) not in self.state['targets'] for t in targets):
            self.validate()
            points=sorted(self.state['points'].values(),key=lambda p:p['angle_deg'])
            pairs=[(p['angle_deg'],p['receiver_ml']) for p in points]
            if any(b[1]<=a[1] for a,b in zip(pairs[:-1],pairs[1:])):
                raise RuntimeError('Raw MPM volume reversal: stop and review; do not sort or smooth it away')
            proposals=[]
            for target in targets:
                if str(target) in self.state['targets']:continue
                best=min(points,key=lambda p:abs(p['receiver_ml']-target))
                if abs(best['receiver_ml']-target)<=self.model['target_simulation_tolerance_ml']:
                    self.state['targets'][str(target)]=dict(target_ml=target,command_angle_deg=best['angle_deg'],
                        tilt_duration_s=round(best['angle_deg']/10,3),predicted_ml=best['receiver_ml'],
                        simulated_target_error_ml=best['receiver_ml']-target,
                        result_path=best['result_path'],result_sha256=best['result_sha256'],
                        review_path=best['review_path'],review_sha256=best['review_sha256'])
                else:
                    angle=linear_proposal(pairs,target)
                    if f'{angle:.2f}' in self.state['points']:
                        raise RuntimeError('Inverse proposal repeats a tested angle without meeting tolerance')
                    proposals.append(dict(target_ml=target,angle_deg=angle))
            self.status('selecting_commands',proposals=proposals)
            if proposals:
                index=len(list(WORK.glob('proposals_*.json')))
                write(WORK/f'proposals_{index:03d}.json',dict(source_simulations=points,
                    proposals=proposals,fluid_validation_outcomes_used=[],method='Piecewise linear inversion of MPM only'))
                self.batch([p['angle_deg'] for p in proposals])

    def package(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.interpolate import PchipInterpolator
        rows=[self.state['targets'][str(t)] for t in self.model['targets_ml']]
        for row in rows:
            stored=self.state['points'][f'{row["command_angle_deg"]:.2f}']
            point=self.read_case(row['command_angle_deg'],stored['kind']=='diagnostic')
            assert point['result_sha256']==row['result_sha256']
            assert abs(point['receiver_ml']-row['target_ml'])<=1.
        points=sorted(self.state['points'].values(),key=lambda p:p['angle_deg'])
        aa=np.array([p['angle_deg'] for p in points]);vv=np.array([p['receiver_ml'] for p in points])
        assert np.all(np.diff(aa)>0) and np.all(np.diff(vv)>0)
        old=ROOT/'out/pour_navier_calibration/philip_pilot_20260907_200'
        old_provenance=json.loads((old/'provenance.json').read_text())
        assert digest(old/'philip_pilot_handoff.zip')==old_provenance['zip_sha256']
        destination=ROOT/'out/pour_navier_calibration/philip_corrected_20260908'
        destination.mkdir(exist_ok=False)
        with (destination/'angle_table.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['target_ml','command_angle_deg','tilt_duration_s'])
            for row in rows:writer.writerow([row['target_ml'],f'{row["command_angle_deg"]:.2f}',f'{row["tilt_duration_s"]:.3f}'])
        fig,ax=plt.subplots(figsize=(9,5.5),constrained_layout=True)
        x=np.linspace(rows[0]['command_angle_deg'],rows[-1]['command_angle_deg'],600)
        ax.plot(x,PchipInterpolator(aa,vv)(x),color='#2166ac',label='MPM prediction')
        selected=np.array([r['command_angle_deg'] for r in rows]);volumes=np.array([r['predicted_ml'] for r in rows])
        ax.scatter(selected,volumes,color='#2166ac',s=32,zorder=3,label='Verified commands')
        for row in rows:ax.annotate(f'{row["target_ml"]} mL',(row['command_angle_deg'],row['predicted_ml']),
            xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        ax.set(xlabel='Commanded pour angle (degrees)',ylabel='Received volume (mL)',
            title='300 mL initial fill · 2 s hold',ylim=(45,215))
        ax.grid(alpha=.2);ax.legend();fig.savefig(destination/'angle_table.png',dpi=170);plt.close(fig)
        notes='''CORRECTED MPM COMMANDS — ROBOT VALIDATION

Fill the source to 300 mL and empty the receiver before each run. Keep the same
cup mounting and robot program used for the supplied joint/action recordings.
Use the angle and tilt-command duration in angle_table.csv. Hold 2.0 seconds
after the tilt acknowledgement, then use a 2.0-second return command.

Record joints, action timestamps, side video and the final received amount for
every target. Keep the cup mounted throughout the session if possible.

Each listed command has a direct MPM check. The viscosity comes from one
60-degree video, with source contact calibrated using that same pour. Earlier
validation amounts were not used to fit these commands. The graph interpolates
simulated volumes. These are predictions for validation; robot accuracy is not
yet established.
'''
        (destination/'START_HERE.txt').write_text(notes)
        names=['angle_table.csv','angle_table.png','START_HERE.txt']
        archive=destination/'philip_corrected_handoff.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name in names:z.write(destination/name,arcname=name)
        with zipfile.ZipFile(archive) as z:
            assert z.namelist()==names and z.testzip() is None
            for name in names:assert z.read(name)==(destination/name).read_bytes()
        write(destination/'provenance.json',dict(model_protocol_sha256=digest(MODEL),predictions=rows,
            simulation_points=points,source_contact=.272,eta_pa_s=self.model['eta_pa_s'],
            original_time_check_passed=False,original_time_difference_ml=self.model['original_time_difference_ml'],
            numerical_acceptance_passed=False,robot_accuracy_established=False,
            release_status='Experimental commands for prospective validation; original 1 mL timestep screen failed at 1.0415 mL',
            calibration_interpretation=self.model['calibration_interpretation'],
            fluid_validation_outcomes_used=[],real_error_forecast_saved=False,
            previous_zip_sha256=old_provenance['zip_sha256'],zip_sha256=digest(archive),
            package_file_sha256={name:digest(destination/name) for name in names},
            controller_sha256=digest(Path(__file__))))
        self.status('package_ready_for_visual_review',archive=str(archive),bytes=archive.stat().st_size)

    def run(self):
        try:
            if self.phase=='critical':
                self.adopt_external();self.solve([60,100])
                self.status('critical_targets_complete')
            else:
                assert all(str(t) in self.state['targets'] for t in [60,100])
                self.batch([60.,65.]);self.solve(self.model['targets_ml']);self.package()
        except Exception as exc:
            self.status('stopped_for_review',error=str(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['critical','full'],required=True)
    args=parser.parse_args()
    WORK.mkdir(exist_ok=True)
    with (WORK/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Search(args.phase).run()


if __name__=='__main__':main()
