"""Reproducible contact-triggered force-amplitude/established-dwell planning using the common A/B fits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pyvista as pv
from scipy.optimize import minimize

from experiments.robotics import x_force_profile_control as control
from experiments.robotics import x_shaping as core
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.plastic_shaping_study import save_json, save_npz

ROOT = core.ROOT
BASE = ROOT / 'out/x_common_identification_study_20260910'
OUT = ROOT / 'out/x_force_profile_study_20260910'
SETTINGS = {'baseline': (64, .00005), 'half_dt': (64, .000025), 'grid80': (80, .00005)}
PROTOCOL = dict(scene=core.CONFIG, control=control.CONTROL,
    initialization_source=str(BASE.relative_to(ROOT)), settings=SETTINGS,
    reference_grid=64, reference_dt=.00005,
    planning='Existing identified-model position-plan openings generate smooth nominal closing motions at <=20 mm/s; new force trajectories are predicted in those identified models',
    execution='Track predicted force histories with nominal velocity feedforward and common normalized PI force feedback; stop at the end of each profile or common 12 mm guard, never at a planned final opening',
    comparison='Identical frozen force/reference-velocity arrays cross-executed in fresh true A/B specimens',
    target='Unchanged 64 x 80 x 24.946587 mm rounded X, 90 mL',
    model_checks='Baseline, finer grid, then fresh baseline repeat before true execution',
    feasibility='No guard stops; relative L2 error of mean outward per-finger force <=15% for each full closing profile; sampled height <=45.5mm',
    scope='Supplied-state identification, ideal finger-force measurements and kinematic tools; stock Franka Hand implementation not established',seed=0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(folder):
    assert not (folder / 'protocol.json').exists(), 'Use a fresh experiment directory'
    folder.mkdir(parents=True, exist_ok=True)
    inputs = folder / 'inputs'; inputs.mkdir()
    for src in (BASE / 'inputs').iterdir():
        if src.is_file(): shutil.copy2(src, inputs / src.name)
    shutil.copytree(BASE / 'validation', folder / 'validation')
    for name in ['target.vtp', 'target.json', 'target_outline_m.npy', 'identification_validation.json', 'observation_isolation.json']:
        shutil.copy2(BASE / name, folder / name)
    shutil.copytree(BASE / 'franka/panda_model_snapshot', folder / 'franka/panda_model_snapshot')
    for material in 'AB':
        path=BASE / f'plans/{material}/selected.json'
        selected=json.loads(path.read_text())
        save_json(inputs / f'initialization_{material}.json',dict(gaps_mm=selected['gaps_mm'],
            law=selected['law'],model=selected['model'],source=str(path),source_sha256=digest(path),
            prior_planning_evaluations=32,scope='Identified-model position plan used only to generate nominal motion and predicted force profiles'))
    save_json(folder / 'protocol.json', PROTOCOL)
    source_paths = list(json.loads((BASE / 'source_sha256.json').read_text()))
    source_paths += ['experiments/robotics/x_force_profile_control.py', 'experiments/robotics/x_force_profile_study.py']
    hashes = {}
    for rel in sorted(set(source_paths)):
        src = ROOT / rel
        dst = folder / 'source_snapshot' / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst); hashes[rel] = digest(src)
    save_json(folder / 'source_sha256.json', hashes)
    save_json(folder / 'input_sha256.json', {str(p.relative_to(folder)): digest(p)
        for p in [*inputs.iterdir(), *folder.glob('target*'), *folder.joinpath('validation').iterdir(),
                  folder / 'identification_validation.json', folder / 'observation_isolation.json'] if p.is_file()})
    save_json(folder / 'robot_input_sha256.json', {str(p.relative_to(folder)): digest(p)
        for p in folder.joinpath('franka/panda_model_snapshot').rglob('*') if p.is_file()})
    (folder / 'packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    (folder / 'commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    (folder / 'gpu_before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))


def check(folder):
    assert json.loads((folder / 'protocol.json').read_text()) == json.loads(json.dumps(PROTOCOL))
    for rel, sha in json.loads((folder / 'source_sha256.json').read_text()).items():
        assert digest(ROOT / rel) == digest(folder / 'source_snapshot' / rel) == sha, rel
    for manifest in ['input_sha256.json', 'robot_input_sha256.json']:
        for rel, sha in json.loads((folder / manifest).read_text()).items():
            assert digest(folder / rel) == sha, rel


def save_profiles(path, profiles):
    fields={}
    for i,p in enumerate(profiles):
        for key,value in p.items():fields[f'{i}_{key}']=np.asarray(value)
    save_npz(path,fields)


def load_profiles(path):
    data=np.load(path)
    return [{key:data[f'{i}_{key}'].copy() for key in ['velocity_ff','feedback_reference','force_reference','peak_n']} for i in range(4)]


def reference(folder,material,device):
    check(folder)
    assert not (folder/f'profiles_{material}.npz').exists()
    initial=json.loads((folder/f'inputs/initialization_{material}.json').read_text())
    profiles=control.nominal_profiles(initial['gaps_mm'])
    cfg=dict(law=initial['law'],gaps_mm=initial['gaps_mm'],grid=64,dt=.00005,device=device,
             scope='Nominal model motion, force feedback disabled; no true-material data',protocol_sha256=digest(folder/'protocol.json'))
    stem=folder/f'reference_{material}'
    save_json(stem.with_suffix('.config.json'),cfg)
    data,phases,stops=control.execute(initial['law'],profiles,64,.00005,device,reference=True)
    assert all(s=='profile_complete' for s in stops)
    np.testing.assert_allclose(data['min_gaps_mm'],initial['gaps_mm'],atol=1e-6)
    save_npz(stem.with_suffix('.npz'),data);save_json(stem.with_suffix('.phases.json'),phases)
    profiles=control.reference_profiles(data)
    save_profiles(folder/f'profiles_{material}.npz',profiles)
    error=mesh_distance_mm(surface(data['x_after_1s'],data['vol0'],h=.00125),pv.read(folder/'target.vtp'))
    record=dict(**cfg,file=stem.with_suffix('.npz').name,surface_mm=error,
        data_sha256=digest(stem.with_suffix('.npz')),profiles_sha256=digest(folder/f'profiles_{material}.npz'),
        peaks_n=[float(p['peak_n']) for p in profiles],durations_s=[len(p['velocity_ff'])*.004 for p in profiles])
    save_json(stem.with_suffix('.json'),record);print(json.dumps(record),flush=True)


def rollout(folder,path,law,planned,device,grid,dt):
    profile_path=folder/f'profiles_{planned}.npz'
    profiles=load_profiles(profile_path)
    config=dict(law=law,planned_for=planned,profiles_sha256=digest(profile_path),
                n_grid=grid,dt=dt,device=device,seed=0,protocol_sha256=digest(folder/'protocol.json'))
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert json.loads(path.with_suffix('.config.json').read_text())==config
        return np.load(path),json.loads(path.with_suffix('.json').read_text())
    save_json(path.with_suffix('.config.json'),config)
    start=time.monotonic()
    data,phases,stops=control.execute(law,profiles,grid,dt,device)
    save_npz(path,data);save_json(path.with_suffix('.phases.json'),phases)
    diag=control.diagnostics(data,phases)
    error=mesh_distance_mm(surface(data['x_after_1s'],data['vol0'],h=.00125),pv.read(folder/'target.vtp'))
    feasible=all(s=='profile_complete' for s in stops) and diag['max_recorded_height_mm']<=45.5 and all(p['force_relative_l2']<=.15 for p in diag['pulses'])
    record=dict(**config,file=str(path.relative_to(folder)),surface_mm=error,stops=stops,feasible=feasible,
        peaks_n=data['peaks_n'].tolist(),durations_s=data['durations_s'].tolist(),
        diagnostics=diag,data_sha256=digest(path),elapsed_s=time.monotonic()-start)
    save_json(path.with_suffix('.json'),record);print(json.dumps(record),flush=True)
    return data,record


def model_checks(folder,material,device):
    check(folder)
    law=json.loads((folder/f'inputs/initialization_{material}.json').read_text())['law']
    records=[];base=None
    for setting,(grid,dt) in {'baseline':(64,.00005),'grid80':(80,.00005),'repeat_baseline':(64,.00005)}.items():
        print(json.dumps(dict(starting=material,setting=setting)),flush=True)
        data,r=rollout(folder,folder/f'model_checks/{material}/{setting}.npz',law,material,device,grid,dt)
        records.append(dict(**r,setting=setting));save_json(folder/f'model_checks/{material}/results.json',records)
        if setting=='baseline':base=(np.asarray(data['x_after_1s']).copy(),r)
        if setting=='repeat_baseline':
            repeat=dict(surface_error_difference_mm=abs(base[1]['surface_mm']-r['surface_mm']),
                        particle_rms_difference_mm=float(np.sqrt(np.mean(np.sum((base[0]-data['x_after_1s'])**2,axis=1)))*1000),
                        same_stops=base[1]['stops']==r['stops'])
            save_json(folder/f'model_checks/{material}/repeat_check.json',repeat)
    print(json.dumps(dict(material=material,all_feasible=all(r['feasible'] for r in records),repeat=repeat)),flush=True)


def freeze(folder):
    check(folder)
    assert not (folder/'execution_plan.json').exists()
    selected={}
    for m in 'AB':
        rows=json.loads((folder/f'model_checks/{m}/results.json').read_text())
        assert len(rows)==3 and all(r['feasible'] for r in rows),m
        repeat=json.loads((folder/f'model_checks/{m}/repeat_check.json').read_text())
        assert repeat['same_stops'] and repeat['surface_error_difference_mm']<.10 and repeat['particle_rms_difference_mm']<.5,repeat
        selected[m]=dict(profiles_sha256=digest(folder/f'profiles_{m}.npz'),reference_sha256=digest(folder/f'reference_{m}.json'),model_checks=rows,repeat=repeat)
    save_json(folder/'execution_plan.json',dict(plans=selected,protocol_sha256=digest(folder/'protocol.json'),scope='Both profile plans frozen before true-material execution'))


def evaluate(folder,material,device,settings):
    check(folder)
    frozen=json.loads((folder/'execution_plan.json').read_text())
    for m in 'AB':assert frozen['plans'][m]['profiles_sha256']==digest(folder/f'profiles_{m}.npz')
    laws=json.loads((folder/'inputs/models.json').read_text())
    for setting in settings:
        grid,dt=SETTINGS[setting]
        for planned in 'AB':
            path=folder/f'{setting}_{material}_plan_{planned}.npz'
            _,r=rollout(folder,path,laws[f'true_{material}'],planned,device,grid,dt)
            save_json(path.with_suffix('.json'),dict(**r,material=material,setting=setting))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','reference','model_checks','freeze','evaluate'])
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--material',choices=['A','B']);p.add_argument('--device',default='cuda:0')
    p.add_argument('--settings',nargs='+',choices=list(SETTINGS),default=list(SETTINGS))
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='reference':reference(a.out,a.material,a.device)
    elif a.stage=='model_checks':model_checks(a.out,a.material,a.device)
    elif a.stage=='freeze':freeze(a.out)
    else:evaluate(a.out,a.material,a.device,a.settings)
