"""Model-only common gain selection followed by frozen force-profile execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pyvista as pv
from experiments.robotics import x_force_profile_study as base
from experiments.robotics import x_force_profile_control as control
from experiments.robotics.plastic_shaping_study import save_json, save_npz
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.hex_shaping_surface import mesh_distance_mm

OUT, SETTINGS, digest, load_profiles = base.OUT, base.SETTINGS, base.digest, base.load_profiles
GAINS = [1., 1.5, 2., 3., .5, .25]
PROTOCOL = dict(candidates=GAINS,
    extension='After stronger feedback worsened B finer-grid tracking and shaping, test common multipliers 0.5 and 0.25 before any true-material execution; prior source/protocol and results preserved in gain_history',
    scope='One common multiplier for both PI gains, both plans and both true materials; model-only selection',
    selection='Among candidates feasible for both identified models on 80^3/50us, minimize worst surface error, breaking ties by lower gain',
    feasibility='Same predeclared profile completion, <=15% force relative L2, <=45.5mm sampled height checks',
    validation='Selected gain receives fresh baseline and repeat checks for both models; freeze before true executions',
    unchanged='Force profiles, nominal velocity profiles, material definitions, target, scoring, travel/speed/acceleration limits')


def prepare(folder):
    base.check(folder)
    assert not (folder/'gain_protocol.json').exists()
    save_json(folder/'gain_protocol.json', PROTOCOL)
    shutil.copy2(__file__, folder/'gain_source.py')


def check(folder):
    base.check(folder)
    assert json.loads((folder/'gain_protocol.json').read_text()) == PROTOCOL
    assert digest(__file__) == digest(folder/'gain_source.py')
    history=folder/'gain_history/sha256.json'
    if history.exists():
        for rel,sha in json.loads(history.read_text()).items():assert digest(history.parent/rel)==sha


def rollout(folder,path,law,planned,device,grid,dt,multiplier):
    cfg=dict(law=law,planned_for=planned,profiles_sha256=digest(folder/f'profiles_{planned}.npz'),
        n_grid=grid,dt=dt,device=device,seed=0,protocol_sha256=digest(folder/'protocol.json'),
        gain_protocol_sha256=digest(folder/'gain_protocol.json'),gain_multiplier=multiplier)
    if path.exists():
        assert json.loads(path.with_suffix('.config.json').read_text()) == cfg
        return np.load(path),json.loads(path.with_suffix('.json').read_text())
    path.parent.mkdir(parents=True,exist_ok=True)
    save_json(path.with_suffix('.config.json'),cfg)
    profiles=load_profiles(folder/f'profiles_{planned}.npz')
    original=control.CONTROL.copy()
    control.CONTROL.update(kp_m_s_n=original['kp_m_s_n']*multiplier,
                           ki_m_s2_n=original['ki_m_s2_n']*multiplier)
    start=time.monotonic()
    try:
        data,phases,stops=control.execute(law,profiles,grid,dt,device)
    finally:
        control.CONTROL.clear();control.CONTROL.update(original)
    data['gain_multiplier']=np.array(multiplier)
    save_npz(path,data);save_json(path.with_suffix('.phases.json'),phases)
    diag=control.diagnostics(data,phases)
    error=mesh_distance_mm(surface(data['x_after_1s'],data['vol0'],h=.00125),pv.read(folder/'target.vtp'))
    feasible=all(s=='profile_complete' for s in stops) and diag['max_recorded_height_mm']<=45.5 and all(p['force_relative_l2']<=.15 for p in diag['pulses'])
    r=dict(**cfg,file=str(path.relative_to(folder)),surface_mm=error,stops=stops,feasible=feasible,
        peaks_n=data['peaks_n'].tolist(),durations_s=data['durations_s'].tolist(),
        diagnostics=diag,data_sha256=digest(path),elapsed_s=time.monotonic()-start)
    save_json(path.with_suffix('.json'),r);print(json.dumps(r),flush=True)
    return data,r


def sweep(folder,material,device):
    check(folder)
    law=json.loads((folder/f'inputs/initialization_{material}.json').read_text())['law']
    original=json.loads((folder/f'model_checks/{material}/grid80.json').read_text())
    result_path=folder/f'gain_checks/{material}/results.json'
    rows=json.loads(result_path.read_text()) if result_path.exists() else [dict(**original,gain_multiplier=1.)]
    for gain in GAINS[1:]:
        if any(r['gain_multiplier']==gain for r in rows):
            continue
        _,r=rollout(folder,folder/f'gain_checks/{material}/gain_{gain:g}.npz',law,material,device,80,.00005,gain)
        rows.append(r)
        save_json(folder/f'gain_checks/{material}/results.json',rows)


def select(folder):
    check(folder)
    assert not (folder/'gain_selected.json').exists()
    rows={m:json.loads((folder/f'gain_checks/{m}/results.json').read_text()) for m in 'AB'}
    assert all(len(r)==len(GAINS) for r in rows.values())
    candidates=[]
    for gain in GAINS:
        pair=[next(r for r in rows[m] if r['gain_multiplier']==gain) for m in 'AB']
        if all(r['feasible'] for r in pair):
            candidates.append(dict(gain_multiplier=gain,worst_surface_mm=max(r['surface_mm'] for r in pair),models=pair))
    assert candidates, 'No gain passes the predefined model checks'
    selected=min(candidates,key=lambda r:(r['worst_surface_mm'],r['gain_multiplier']))
    save_json(folder/'gain_selected.json',selected)
    print(json.dumps(selected),flush=True)


def model_checks(folder,material,device):
    check(folder)
    selected=json.loads((folder/'gain_selected.json').read_text())
    gain=selected['gain_multiplier']
    law=json.loads((folder/f'inputs/initialization_{material}.json').read_text())['law']
    rows=[next(r for r in selected['models'] if r['planned_for']==material)]
    for setting in ['baseline','repeat_baseline']:
        data,r=rollout(folder,folder/f'gain_checks/{material}/{setting}.npz',law,material,device,64,.00005,gain)
        rows.append(r)
        if setting=='baseline':first=(np.array(data['x_after_1s']),r)
        else:
            repeat=dict(surface_error_difference_mm=abs(first[1]['surface_mm']-r['surface_mm']),
                particle_rms_difference_mm=float(np.sqrt(np.mean(np.sum((first[0]-data['x_after_1s'])**2,axis=1)))*1000),
                same_stops=first[1]['stops']==r['stops'])
            save_json(folder/f'gain_checks/{material}/repeat_check.json',repeat)
    save_json(folder/f'gain_checks/{material}/selected_checks.json',rows)


def freeze(folder):
    check(folder)
    assert not (folder/'execution_plan.json').exists()
    selected=json.loads((folder/'gain_selected.json').read_text())
    plans={}
    for m in 'AB':
        rows=json.loads((folder/f'gain_checks/{m}/selected_checks.json').read_text())
        assert len(rows)==3 and all(r['feasible'] for r in rows),m
        repeat=json.loads((folder/f'gain_checks/{m}/repeat_check.json').read_text())
        assert repeat['same_stops'] and repeat['surface_error_difference_mm']<.1 and repeat['particle_rms_difference_mm']<.5
        plans[m]=dict(profiles_sha256=digest(folder/f'profiles_{m}.npz'),reference_sha256=digest(folder/f'reference_{m}.json'),model_checks=rows,repeat=repeat)
    save_json(folder/'execution_plan.json',dict(plans=plans,gain_multiplier=selected['gain_multiplier'],
        gain_selected_sha256=digest(folder/'gain_selected.json'),protocol_sha256=digest(folder/'protocol.json'),
        gain_protocol_sha256=digest(folder/'gain_protocol.json'),scope='Both profiles and one common gain frozen before true-material execution'))


def evaluate(folder,material,device,settings):
    check(folder)
    frozen=json.loads((folder/'execution_plan.json').read_text())
    assert frozen['gain_selected_sha256']==digest(folder/'gain_selected.json')
    for m in 'AB':assert frozen['plans'][m]['profiles_sha256']==digest(folder/f'profiles_{m}.npz')
    laws=json.loads((folder/'inputs/models.json').read_text())
    for setting in settings:
        for planned in 'AB':
            grid,dt=SETTINGS[setting]
            path=folder/f'{setting}_{material}_plan_{planned}.npz'
            _,r=rollout(folder,path,laws[f'true_{material}'],planned,device,grid,dt,frozen['gain_multiplier'])
            save_json(path.with_suffix('.json'),dict(**r,material=material,setting=setting))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','sweep','select','model_checks','freeze','evaluate'])
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--material',choices=['A','B']);p.add_argument('--device',default='cuda:0')
    p.add_argument('--settings',nargs='+',choices=list(SETTINGS),default=list(SETTINGS))
    a=p.parse_args()
    if a.stage in ['prepare','select','freeze']:globals()[a.stage](a.out)
    elif a.stage=='evaluate':evaluate(a.out,a.material,a.device,a.settings)
    else:globals()[a.stage](a.out,a.material,a.device)
