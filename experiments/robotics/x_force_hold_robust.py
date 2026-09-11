"""Model-only paired force/dwell refinement over two numerical resolutions.

This additional stage follows the interrupted single-grid local search. It
preserves every earlier result and records its own protocol before simulation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from experiments.robotics import x_force_hold_plan as planning
from experiments.robotics.plastic_shaping_study import save_json

PROTOCOL=dict(
    initialization='Lowest feasible complete initialization surface error, identified model only',
    settings={'planning64':[64,.0001], 'grid80':[80,.00005]},
    variables='Coupled force/dwell changes for the first two pinches; last two commands initially frozen',
    seed_indices=[0.,1.,2.], force_factor='1 + 0.025*u per pinch', dwell_factor='1 - 0.35*u per pinch, with 80 ms minimum',
    seed_candidates=9,
    local_refinement='Four neighbors of best feasible seed, one index at a time +/-0.5, clipped to [0,2.5]',
    fallback='If no feasible seed, take up to three guard-free seeds with lowest worst shape error; test last hold >=0.4s, then last force multiplied by 0.9 with that longer hold',
    feasibility='All dwells complete, each hold raw mean-finger force RMSE <=15% of command, sampled height <=45.5mm, in both settings',
    selection='Lowest worst-case identified-model shape error, followed by 64^3/50us check and fresh baseline repeat before true execution',
    scope='No true-material shaping results used; no changes to materials, target, controller, or error metric')


def prepare(folder):
    planning.stage(folder)
    dest=folder/'robust';dest.mkdir(exist_ok=True)
    assert not (dest/'protocol.json').exists()
    save_json(dest/'protocol.json',PROTOCOL);shutil.copy2(__file__,dest/'source.py')
    counts={m:len(json.loads((folder/f'plans/{m}/evaluations.json').read_text())) for m in 'AB'}
    save_json(folder/'single_grid_search_stopped.json',dict(completed_objective_calls=counts,planned_budget_per_material=48,
        reason='Early identified-model grid check exposed shape sensitivity and a force-tracking failure; stopped single-grid refinement to spend evaluations on paired force/dwell changes assessed on both grids',true_executions_at_stop=0))
    for m in 'AB':
        rows=json.loads((folder/f'plans/{m}/initialization_evaluations.json').read_text())
        best=min((r for r in rows if r['feasible'] and len(r['peaks_n'])==4),key=lambda r:r['surface_mm'])
        save_json(dest/f'initial_{m}.json',best)


def check(folder):
    planning.stage(folder)
    assert json.loads((folder/'robust/protocol.json').read_text())==PROTOCOL
    assert planning.study.digest(__file__)==planning.study.digest(folder/'robust/source.py')


def run(folder, material, device):
    check(folder)
    dest=folder/'robust'/material;dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'completed.json').exists()
    initial=json.loads((folder/f'robust/initial_{material}.json').read_text())
    law=json.loads((folder/f'inputs/initialization_{material}.json').read_text())['law']
    rows=[]
    def trial(u, label, late=None):
        force=np.array(initial['peaks_n']);dwell=np.array(initial['durations_s'])
        force[:2]*=1+.025*np.array(u)
        dwell[:2]=np.maximum(.08,dwell[:2]*(1-.35*np.array(u)))
        if late is not None:
            dwell[-1]=max(dwell[-1],.4)
            force[-1]*=late
        print(json.dumps(dict(starting=material,label=label,u=list(u),late_factor=late)),flush=True)
        checks=[]
        for name,(grid,dt) in PROTOCOL['settings'].items():
            try:
                _,r=planning.study.rollout(folder,dest/name,law,force,dwell,device,grid,dt)
                checks.append(dict(**r,setting=name))
                print(json.dumps(dict(material=material,label=label,setting=name,error_mm=r['surface_mm'],feasible=r['feasible'],stops=r['stops'],hold_rmse=[p['relative_hold_rmse'] for p in r['diagnostics']['pulses']])),flush=True)
            except (RuntimeError,ValueError,FloatingPointError) as exc:
                checks.append(dict(setting=name,feasible=False,surface_mm=1000.,failure=str(exc),stops=['failure']))
        row=dict(label=label,u=list(u),late_force_factor=late,checks=checks,
                 feasible=all(r['feasible'] for r in checks),worst_error_mm=max(r['surface_mm'] for r in checks))
        rows.append(row);save_json(dest/'evaluations.json',rows)
        return row
    for u in ((a,b) for a in PROTOCOL['seed_indices'] for b in PROTOCOL['seed_indices']):
        trial(u,f'seed_{len(rows)+1}')
    feasible=[r for r in rows if r['feasible']]
    if not feasible:
        candidates=sorted((r for r in rows if all(all(s=='dwell_complete' for s in q['stops']) for q in r['checks'])),key=lambda r:r['worst_error_mm'])[:3]
        for r in candidates:
            for factor in [1.,.9]: trial(r['u'],f'fallback_{len(rows)+1}',factor)
        feasible=[r for r in rows if r['feasible']]
    if feasible:
        best=min(feasible,key=lambda r:r['worst_error_mm'])
        for j,delta in [(0,-.5),(0,.5),(1,-.5),(1,.5)]:
            u=np.array(best['u']);u[j]=np.clip(u[j]+delta,0,2.5)
            trial(u,f'neighbor_{len(rows)+1}',best['late_force_factor'])
    save_json(dest/'completed.json',dict(evaluations=len(rows),feasible=sum(r['feasible'] for r in rows),
        protocol_sha256=planning.study.digest(folder/'robust/protocol.json')))
    print(json.dumps(dict(material=material,completed=len(rows),feasible=sum(r['feasible'] for r in rows))),flush=True)


def select(folder,material,device):
    check(folder)
    dest=folder/'robust'/material
    assert (dest/'completed.json').exists()
    assert not (folder/f'plans/{material}/selection_complete.json').exists()
    rows=json.loads((dest/'evaluations.json').read_text())
    feasible=sorted((r for r in rows if r['feasible']),key=lambda r:r['worst_error_mm'])
    assert feasible, 'No robust model-feasible candidate'
    checked=[]
    for candidate in feasible[:3]:
        r=candidate['checks'][0]
        d,b=planning.study.rollout(folder,dest/'baseline',r['law'],r['peaks_n'],r['durations_s'],device,64,.00005)
        checked.append(dict(candidate=candidate,baseline=b,worst_error_mm=max(candidate['worst_error_mm'],b['surface_mm'])))
    save_json(dest/'baseline_checks.json',checked)
    candidates=[r for r in checked if r['baseline']['feasible']]
    assert candidates,'No candidate passes baseline model check'
    chosen=min(candidates,key=lambda r:r['worst_error_mm']);b=chosen['baseline']
    before=np.load(folder/b['file'])
    after,repeat=planning.study.rollout(folder,dest/'repeat_baseline',b['law'],b['peaks_n'],b['durations_s'],device,64,.00005)
    repeat_check=dict(surface_error_difference_mm=abs(b['surface_mm']-repeat['surface_mm']),
        particle_rms_difference_mm=float(np.sqrt(np.mean(np.sum((before['x_after_1s']-after['x_after_1s'])**2,axis=-1)))*1000),
        feasible=repeat['feasible'],same_stops=b['stops']==repeat['stops'],file=repeat['file'])
    save_json(dest/'repeat_check.json',repeat_check)
    assert repeat_check['feasible'] and repeat_check['same_stops']
    assert repeat_check['surface_error_difference_mm']<.10 and repeat_check['particle_rms_difference_mm']<.5,repeat_check
    model=json.loads((folder/f'inputs/initialization_{material}.json').read_text())['model']
    selected=dict(**b,model=model,worst_model_error_mm=chosen['worst_error_mm'],
        robust_candidate=chosen['candidate'],baseline_checks=len(checked),repeat_check=repeat_check,
        selection_protocol_sha256=planning.study.digest(folder/'robust/protocol.json'))
    path=folder/f'plans/{material}/selected.json';save_json(path,selected)
    save_json(path.with_name('selection_complete.json'),dict(selected_sha256=planning.study.digest(path),record=selected))
    print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run','select'])
    p.add_argument('--out',type=Path,default=planning.study.OUT);p.add_argument('--material',choices=['A','B']);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='run':run(a.out,a.material,a.device)
    else:select(a.out,a.material,a.device)
