"""Select normalized-force plans from identified-model checks before true execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from experiments.robotics import x_force_hold_damped_plan as planning
from experiments.robotics import x_force_hold_robust as earlier
from experiments.robotics.plastic_shaping_study import save_json

OLD=planning.study.ROOT/'out/x_force_hold_study_20260910'
PROTOCOL=dict(
    candidates=3,
    A='Three distinct lowest-worst-error feasible commands from the completed earlier A two-grid refinement, restricted to forces <=1.25N so normalization leaves its controller unchanged',
    B='Three distinct lowest-error feasible full commands from the normalized-controller B initialization and 24-call local search',
    settings={'baseline':[64,.00005],'grid80':[80,.00005]},
    selection='Lowest worst-case identified-model surface error over planning and both validation settings; all holds completed, every raw hold force RMS error <=15%, sampled height <=45.5mm',
    repeat='Fresh selected baseline model repeat, same stops, surface-error difference <0.10mm and particle RMS difference <0.5mm',
    scope='No true-material outcomes used; A is carried over only because the common command-dependent gain multiplier is exactly one at every command')


def prepare(folder):
    planning.stage(folder);earlier.check(OLD)
    assert (OLD/'robust/A/completed.json').exists()
    assert not (folder/'selection_protocol.json').exists()
    save_json(folder/'selection_protocol.json',PROTOCOL);shutil.copy2(__file__,folder/'selection_source.py')
    src=folder/'selection_inputs/A';src.mkdir(parents=True,exist_ok=True)
    for name,path in dict(evaluations=OLD/'robust/A/evaluations.json',completion=OLD/'robust/A/completed.json',
                          protocol=OLD/'robust/protocol.json',initial=OLD/'robust/initial_A.json').items():
        shutil.copy2(path,src/f'{name}.json')
    save_json(folder/'selection_input_sha256.json',{str(p.relative_to(folder)):planning.study.digest(p) for p in src.iterdir()})


def check(folder):
    planning.stage(folder)
    assert json.loads((folder/'selection_protocol.json').read_text())==PROTOCOL
    assert planning.study.digest(__file__)==planning.study.digest(folder/'selection_source.py')
    for rel,sha in json.loads((folder/'selection_input_sha256.json').read_text()).items():assert planning.study.digest(folder/rel)==sha


def select(folder,material,device):
    check(folder);dest=folder/'plans'/material;dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'selection_complete.json').exists()
    if material=='A':
        raw=json.loads((folder/'selection_inputs/A/evaluations.json').read_text())
        rows=[dict(**r['checks'][0],selection_rank_error=r['worst_error_mm'],earlier_candidate=r)
              for r in raw if r['feasible'] and max(r['checks'][0]['peaks_n'])<=1.25]
    else:
        assert (dest/'selected.json').exists(), 'Complete the 24-call B search first'
        shutil.copy2(dest/'selected.json',dest/'search_selected.json')
        raw=json.loads((dest/'evaluations.json').read_text())+json.loads((dest/'initialization_evaluations.json').read_text())
        rows=[dict(**r,selection_rank_error=r['surface_mm']) for r in raw if r['feasible'] and len(r['peaks_n'])==4]
    unique={}
    for r in rows:unique[(tuple(r['peaks_n']),tuple(r['durations_s']))]=r
    candidates=sorted(unique.values(),key=lambda r:r['selection_rank_error'])[:PROTOCOL['candidates']]
    law=json.loads((folder/f'inputs/initialization_{material}.json').read_text())['law']
    checks=[]
    for i,candidate in enumerate(candidates):
        results=[]
        for setting,(grid,dt) in PROTOCOL['settings'].items():
            print(json.dumps(dict(starting=material,candidate=i+1,setting=setting)),flush=True)
            _,r=planning.study.rollout(folder,folder/'model_checks'/material/setting,law,candidate['peaks_n'],candidate['durations_s'],device,grid,dt)
            results.append(dict(**r,setting=setting))
            print(json.dumps(dict(material=material,candidate=i+1,setting=setting,surface_mm=r['surface_mm'],feasible=r['feasible'],stops=r['stops'])),flush=True)
        checks.append(dict(planning=candidate,checks=results,feasible=all(r['feasible'] for r in results),
                           worst_error_mm=max(candidate['selection_rank_error'],*[r['surface_mm'] for r in results])))
        save_json(dest/'model_checks.json',checks)
    feasible=[r for r in checks if r['feasible']]
    assert feasible,'No model-feasible normalized-force plan across numerical settings'
    chosen=min(feasible,key=lambda r:r['worst_error_mm'])
    baseline=next(r for r in chosen['checks'] if r['setting']=='baseline')
    before=np.load(folder/baseline['file'])
    after,repeat=planning.study.rollout(folder,folder/'model_checks'/material/'repeat_baseline',law,baseline['peaks_n'],baseline['durations_s'],device,64,.00005)
    repeat_check=dict(surface_error_difference_mm=abs(baseline['surface_mm']-repeat['surface_mm']),
        particle_rms_difference_mm=float(np.sqrt(np.mean(np.sum((before['x_after_1s']-after['x_after_1s'])**2,axis=-1)))*1000),
        feasible=repeat['feasible'],same_stops=baseline['stops']==repeat['stops'],file=repeat['file'])
    save_json(dest/'repeat_check.json',repeat_check)
    assert repeat_check['feasible'] and repeat_check['same_stops']
    assert repeat_check['surface_error_difference_mm']<.10 and repeat_check['particle_rms_difference_mm']<.5,repeat_check
    initial=json.loads((folder/f'inputs/initialization_{material}.json').read_text())
    result=dict(**baseline,model=initial['model'],worst_model_error_mm=chosen['worst_error_mm'],
                selection_candidate=chosen,repeat_check=repeat_check,
                selection_protocol_sha256=planning.study.digest(folder/'selection_protocol.json'))
    save_json(dest/'selected.json',result)
    save_json(dest/'selection_complete.json',dict(selected_sha256=planning.study.digest(dest/'selected.json'),record=result))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','select'])
    p.add_argument('--out',type=Path,default=planning.study.OUT);p.add_argument('--material',choices=['A','B']);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.out)
    else:select(a.out,a.material,a.device)
