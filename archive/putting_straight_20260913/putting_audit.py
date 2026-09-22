"""Summarize every reported execution and preserve a reproducible experiment bundle."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import importlib.metadata
import numpy as np
from experiments.elastic.putting import ROOT,save,digest
from experiments.elastic.putting_franka import load_motion

def read(path):return json.loads(Path(path).read_text())

def audit_case(path):
    result=read(path/'result.json');config=read(path/'config.json');p=config['protocol']
    provenance=read(path/'execution_provenance.json');motion_path=Path(provenance['motion'])
    assert digest(motion_path)==provenance['motion_sha256']
    motion=load_motion(motion_path);data=np.load(path/'trajectory.npz');s=data['signals'];x=data['x'];ref=data['reference']
    assert np.isfinite(x).all() and np.isfinite(s).all()
    if len(x):assert result['runout_s']==0, 'Video must include the ball coming to rest, without freezing the club.'
    all_ball=np.vstack([s[:,:10],data['runout']]) if len(data['runout']) else s[:,:10]
    target=np.array(p['target']);error=np.linalg.norm(np.array(result['final_ball_m'])[:2]-target)*1000
    np.testing.assert_allclose(error,result['error_mm'],atol=1e-8)
    # A short numerical check can stop between saved 10 ms samples. During
    # free runout the speed decreases, so the last speed bounds this distance.
    lag=s[-1,0]+result['runout_s']-all_ball[-1,0]
    saved_gap=np.linalg.norm(all_ball[-1,1:4]-result['final_ball_m'])
    assert -.0000001<=lag<=p['video_dt']+.0000001
    assert saved_gap<=np.linalg.norm(all_ball[-1,4:7])*max(lag,0)+1e-8
    footprint=p['ball_radius'];b=p['table_bounds']
    assert (all_ball[:,1]>b[0]+footprint).all() and (all_ball[:,1]<b[1]-footprint).all()
    assert (all_ball[:,2]>b[2]+footprint).all() and (all_ball[:,2]<b[3]-footprint).all()
    core=(abs(ref[:,2])<.003)&(abs(ref[:,0])<.003)&(abs(ref[:,1])<.002)
    errors=[]
    for si,xi in zip(s,x):
        c,R=motion(si[0]);errors.append(np.linalg.norm(xi[core]-(ref[core]@R.T+c),axis=1).max())
    moving=np.flatnonzero(np.linalg.norm(all_ball[:,4:7],axis=1)>1e-5)
    result.update(max_grasp_core_pose_error_mm=float(max(errors)*1000) if errors else None,
        last_saved_ball_position_lag_mm=float(saved_gap*1000),
        stopping_time_s=float(all_ball[moving[-1],0]) if len(moving) else 0.,
        numerics=dict(grid=p['grid'],dt=p['dt'],coupling_dt=p['coupling_dt']),
        case_sha256={name:digest(path/name) for name in ['config.json','result.json','execution_provenance.json']})
    return result

def summarize(root,prefix='execution'):
    p=read(root/'protocol.json');report=dict(protocol=p,plans={},executions={},numerical_checks={})
    for k in 'AB':
        prov=read(root/f'{prefix}_A_plan_{k}/execution_provenance.json')
        report['plans'][k]=read(prov['plan_path'])
    for folder in sorted(root.glob('*_?_plan_?')):
        if not (folder/'execution_provenance.json').exists() or not (folder/'result.json').exists():continue
        m,k=folder.name.split('_')[-3],folder.name.split('_')[-1]
        cfg=read(folder/'config.json');prov=read(folder/'execution_provenance.json');plan=read(prov['plan_path'])
        assert digest(prov['plan_path'])==prov['plan_sha256']
        assert cfg['E_pa']==p['E_true'][m] and cfg['controls']==plan['controls']
        assert prov['plan_material']==k and prov['executed_material']==m
        for key,value in p.items():
            if key not in ['grid','dt','coupling_dt','end','video_dt']:assert cfg['protocol'][key]==value
        group='executions' if folder.name.startswith(prefix+'_') else 'numerical_checks'
        report[group][folder.name]=audit_case(folder)
    for k in 'AB':
        pa=read(root/f'{prefix}_A_plan_{k}/execution_provenance.json')
        pb=read(root/f'{prefix}_B_plan_{k}/execution_provenance.json')
        assert pa['motion_sha256']==pb['motion_sha256'] and pa['plan_sha256']==pb['plan_sha256']
    save(root/'report.json',report)
    dest=root/'final_sources';dest.mkdir(exist_ok=True)
    sources=list((ROOT/'experiments/elastic').glob('putting*.py'))+[ROOT/'tests/test_putting.py']
    for file in sources:shutil.copy2(file,dest/file.name)
    relevant=sources+[ROOT/f'experiments/elastic/{name}.py' for name in ['rod_insertion_franka','rod_insertion_study','block_drop_study','block_drop_report']]+list((ROOT/'src/warpmpm').rglob('*.py'))
    save(root/'reproduction_source_hashes.json',{str(f.relative_to(ROOT)):digest(f) for f in relevant})
    versions={}
    for package in ['numpy','scipy','warp-lang','mujoco','pyvista','vtk','imageio','imageio-ffmpeg','scikit-image','Pillow','torch']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='unknown'
    save(root/'environment.json',dict(python=sys.version,packages=versions,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        gpu_and_driver=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        new_sources='New putting modules are saved in final_sources; the git revision alone does not contain this experiment.'))
    lines=['# Elastic-club putting: simulation report','',
        'The robot makes one prescribed stroke with a flexible club. The ball must come to rest entirely inside a marked circle, 612 mm from its starting center. The circle radius is 35 mm; the ball radius is 8 mm, so success requires a center error of at most 27 mm.','',
        '| True material | Plan using matched ID | Plan using swapped ID |',
        '|---|---:|---:|']
    for m in 'AB':
        a=report['executions'][f'{prefix}_{m}_plan_{m}'];other='B' if m=='A' else 'A';b=report['executions'][f'{prefix}_{m}_plan_{other}']
        def value(v):return f"{v['error_mm']:.1f} mm ({'inside' if v['success'] else 'outside'})"
        lines.append(f'| {m} | {value(a)} | {value(b)} |')
    fine_matches=[report['numerical_checks'].get(f'fine_{m}_plan_{m}') for m in 'AB']
    if all(r is not None for r in fine_matches):
        lines+=['',f"**Precision qualification.** On the finer grid, matched errors are {fine_matches[0]['error_mm']:.1f} mm (A) and {fine_matches[1]['error_mm']:.1f} mm (B). The smaller errors in the main video are resolution-specific, not a demonstrated continuum-accuracy claim. Further contact/grid convergence work is needed before claiming millimeter precision."]
    lines+=['','**Control.** Optimize two continuous parameters: stroke duration and aim angle. Travel is fixed at 65 mm, with a smooth quintic motion followed by withdrawal. The complete Franka command is frozen and reused on both materials. There is no force, shape, or ball-position feedback.','',
        '| Plan | Identified stiffness | Stroke duration | Aim | Simulations in selected search |',
        '|---|---:|---:|---:|---:|']
    for k,pl in report['plans'].items():lines.append(f"| {k} | {pl['E_pa']/1000:.2f} kPa | {pl['controls'][0]:.4f} s | {pl['controls'][1]:.2f}° | {pl['evaluations']} |")
    lines+=['','**Identification inputs.** These are the existing textured-stereo bending estimates: A = 80.61 kPa and B = 248.11 kPa, against true stiffnesses of 80 and 240 kPa. They were not fitted to putting outcomes. The same homogeneous fixed-corotated material model is used in the new club geometry. The original bending reconstruction assumptions still apply.','',
        '**Numerical checks.**','',
        'The displayed executions use a 1.0 mm MPM grid and 125 μs coupling updates. The fine check uses a 0.75 mm grid; the contact-update check uses 62.5 μs updates on the 1.0 mm grid. The frozen commands are identical in every check.','',
        '| Check | True material / plan | Center error | Change from displayed execution |',
        '|---|---|---:|---:|']
    for name,r in report['numerical_checks'].items():
        parts=name.split('_');m,k=parts[-3],parts[-1];base=report['executions'][f'{prefix}_{m}_plan_{k}']
        change=np.linalg.norm(np.array(r['final_ball_m'])-base['final_ball_m'])*1000
        lines.append(f"| {name.rsplit('_',3)[0]} | {m} / {k} | {r['error_mm']:.1f} mm | {change:.1f} mm |")
    if all(f'{pre}_{m}_plan_{k}' in report['numerical_checks'] for pre in ['fine','half_contact'] for m in 'AB' for k in 'AB'):
        consistent=all(report['numerical_checks'][f'{pre}_{m}_plan_{k}']['success']==(m==k) for pre in ['fine','half_contact'] for m in 'AB' for k in 'AB')
        lines+=['',f"The matched-success / swapped-failure pattern {'holds' if consistent else 'does not hold'} in both numerical checks. This is a task-outcome comparison, not a convergence claim for the exact stopping positions."]
    if 'repeat_fine_A_plan_A' in report['numerical_checks'] and 'fine_A_plan_A' in report['numerical_checks']:
        delta=np.linalg.norm(np.array(report['numerical_checks']['repeat_fine_A_plan_A']['final_ball_m'])-report['numerical_checks']['fine_A_plan_A']['final_ball_m'])*1000
        lines+=['',f'An independent repeat of the finer-grid A matched case changes the stopping point by {delta:.3f} mm.']
    lines+=['','The initial hanging state is prepared by numerical dynamic relaxation. That damping is disabled before the stroke, and elastic deformation is retained. Every resolution uses the same integrated club volume. The rigid-ball model passes independent rolling-distance and sliding-to-rolling checks.','',
        '**Scope.** This is a simulation experiment, with kinematic Franka motion and a bonded grasp. Ball mass, geometry, floor friction and rolling resistance are prescribed and known. Club–ball contact is frictionless; no material damping is identified. Robot torque limits, grasp slip, uncertain surface mechanics, and hardware transfer have not been validated.','',
        '**Development record.** A short 184 mm putt showed little material contrast. A longer target was chosen during task development. Earlier fast-shot pilots and partial searches are retained in the parent directory, including runs superseded after correcting gravity startup oscillations and sampled-volume differences across grids. No swapped-case error enters the planning objective.','',
        '**Videos.** [Comparison](media/putting_comparison.mp4), [contact close-up](media/putting_contact_closeup.mp4), [full Franka view](media/franka_putting.mp4). The main video runs at real time; the contact view is 4× slower. Surfaces are interpolated from saved MPM particles; the ball stripe follows its integrated angular velocity.','',
        '**Reproducibility.** Protocol, identification JSON files, every completed planning trial, frozen commands, robot assets, source snapshots, environment versions, and numerical checks are saved here. The manuscript and its PDF are unchanged.','']
    lines+=['To repeat both searches, all four executions, numerical checks, and videos from the frozen inputs:', '',
        '```bash',f'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.putting_reproduce --source {root.resolve()} --out out/putting_repeat',
        '```','', 'This command requires the recorded source version and two CUDA devices. GPU reductions can produce small floating-point differences; source hashes and the protocol are checked, but bitwise replay is not assumed.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prefix',default='execution')
    a=ap.parse_args();summarize(a.out,a.prefix)
